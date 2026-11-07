"""Move clips to an archive folder after a few days (e.g. a NAS over NFS).

Recording stays local: fast, and it keeps working when the NAS is off. An event only leaves
the local disk once its copy in the archive has the same size; when the archive is not
reachable, everything stays local until it is back. The archive has its own retention.

Once a day, what the service has learned (faces, plates, parking spots, masks, settings) also
goes to <archive>/backup as a tar.gz; the last BACKUP_KEEP are kept.
"""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import threading
import time
from datetime import date
from pathlib import Path
from typing import Callable

from .retention import _event_groups, cleanup

log = logging.getLogger(__name__)

BACKUP_KEEP = 14
# crash marker, GPU cache; auth.json/pairing_code hold the session secret and do not belong on a
# share (after a restore, pair a passkey again)
BACKUP_SKIP = {"running", "intel", "auth.json", "pairing_code"}


class Archive(threading.Thread):
    def __init__(self, clips_dir: Path, archive_dir: Path, local_hours: float, max_days: float,
                 max_gb: float, on_change: Callable[[], None], interval: float = 600,
                 days_by_label: dict[str, float] | None = None, require_mount: bool = True,
                 state_dir: Path | None = None):
        super().__init__(name="archive", daemon=True)
        self.clips_dir, self.dir = clips_dir, archive_dir
        self.local_hours, self.max_days, self.max_gb = local_hours, max_days, max_gb
        self.days_by_label = days_by_label or {}
        self.on_change = on_change
        self.interval = interval
        self.require_mount = require_mount
        self.state_dir = state_dir
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

    def backup_state(self) -> Path | None:
        """No backup of state_dir in the archive today yet: make one."""
        if self.state_dir is None:
            return None
        d = self.dir / "backup"
        dst = d / f"state-{date.today():%Y-%m-%d}.tar.gz"
        if dst.exists():
            return None
        d.mkdir(exist_ok=True)
        tmp = d / (dst.name + ".tmp")
        with tarfile.open(tmp, "w:gz") as tar:
            for p in sorted(self.state_dir.iterdir()):
                # no hidden folders (~/.cache with models) and no temporary files
                if not p.name.startswith(".") and p.name not in BACKUP_SKIP and p.suffix != ".tmp":
                    tar.add(p, arcname=p.name)
        tmp.replace(dst)
        for old in sorted(d.glob("state-*.tar.gz"))[:-BACKUP_KEEP]:
            old.unlink(missing_ok=True)
        log.info("Backup of %s to %s (%.0f kB)", self.state_dir, dst, dst.stat().st_size / 1e3)
        return dst

    def cycle(self) -> None:
        if not self.check():
            return
        try:
            self.backup_state()
        except OSError:
            log.exception("Backup to the archive failed")
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
