"""Move clips to an archive folder after a few days (e.g. a NAS over NFS).

Recording stays local: fast, and it keeps working when the NAS is off. An event only leaves
the local disk once its copy in the archive has the same size; when the archive is not
reachable, everything stays local until it is back. The archive has its own retention.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Callable

from .retention import _event_groups, cleanup

log = logging.getLogger(__name__)


class Archive(threading.Thread):
    def __init__(self, clips_dir: Path, archive_dir: Path, local_hours: float, max_days: float,
                 max_gb: float, on_change: Callable[[], None], interval: float = 600,
                 days_by_label: dict[str, float] | None = None, require_mount: bool = True):
        super().__init__(name="archive", daemon=True)
        self.clips_dir, self.dir = clips_dir, archive_dir
        self.local_hours, self.max_days, self.max_gb = local_hours, max_days, max_gb
        self.days_by_label = days_by_label or {}
        self.on_change = on_change
        self.interval = interval
        self.require_mount = require_mount
        self.ok = False
        self.used_gb = 0.0
        self._stop = threading.Event()

    def check(self) -> bool:
        """Only a real mount counts: otherwise the 'archive' would end up on the local disk."""
        try:
            self.ok = ((os.path.ismount(self.dir) or not self.require_mount)
                       and self.dir.is_dir() and os.access(self.dir, os.W_OK))
        except OSError:
            self.ok = False
        return self.ok

    def move_old(self) -> int:
        """Events older than local_hours to the archive; returns how many were moved."""
        cutoff = time.time() - self.local_hours * 3600
        moved = 0
        for mtime, files in _event_groups(self.clips_dir):   # oldest first
            if mtime >= cutoff:
                break
            if any(".parts" in f.parts for f in files):
                continue
            # JSON last: the web page finds an event through its JSON
            for f in sorted(files, key=lambda p: p.suffix == ".json"):
                try:
                    self._move(f)
                except FileNotFoundError:   # cleaned up by retention in the meantime
                    continue
            moved += 1
        return moved

    def _move(self, src: Path) -> None:
        dst = self.dir / src.relative_to(self.clips_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        st = src.stat()
        shutil.copyfile(src, tmp)
        if tmp.stat().st_size != st.st_size:
            tmp.unlink(missing_ok=True)
            raise OSError(f"copy of {src.name} incomplete")
        os.utime(tmp, (st.st_atime, st.st_mtime))   # retention counts from the recording
        tmp.replace(dst)
        src.unlink()

    def cycle(self) -> None:
        if not self.check():
            return
        moved = self.move_old()
        removed, self.used_gb = cleanup(self.dir, self.max_days, self.max_gb, self.days_by_label)
        if moved or removed:
            log.info("Archive: %d events moved, %d removed, %.2f GB in use",
                     moved, removed, self.used_gb)
            self.on_change()

    def run(self) -> None:
        was_ok = None
        # know reachability right away, but run the move half an interval after the cleanup
        # (retention), so they never work in the same folders at the same time
        self.check()
        if self._stop.wait(self.interval / 2):
            return
        while True:
            try:
                self.cycle()
            except Exception:  # noqa: BLE001
                self.ok = False
                log.exception("Archiving failed; clips stay local")
            if self.ok != was_ok:
                (log.info if self.ok else log.warning)("Archive %s: %s", self.dir,
                                                       "reachable" if self.ok else "not reachable")
                was_ok = self.ok
            if self._stop.wait(self.interval):
                return

    def stop(self) -> None:
        self._stop.set()
