"""Oude clips opruimen: eerst op leeftijd, daarna op totale grootte (oudste eerst)."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

EXTS = (".mp4", ".jpg", ".json")


def _event_groups(clips_dir: Path) -> list[tuple[float, list[Path]]]:
    """Alle events als (mtime, [bestanden]) — mp4/jpg/json met dezelfde naam horen bij elkaar."""
    groups: dict[Path, list[Path]] = {}
    for f in clips_dir.rglob("*"):
        if f.is_file() and f.suffix in EXTS:
            groups.setdefault(f.with_suffix(""), []).append(f)
    out = []
    for files in groups.values():
        try:
            mtime = min(f.stat().st_mtime for f in files)
        except FileNotFoundError:
            continue
        out.append((mtime, files))
    out.sort(key=lambda g: g[0])
    return out


def _label(files: list[Path]) -> str | None:
    """Label uit de bestandsnaam: <tijd>_<label>[_n].<ext> (ook .face1.jpg, .plate1.jpg)."""
    parts = files[0].name.split(".")[0].split("_")
    return parts[1] if len(parts) > 1 else None


def cleanup(clips_dir: Path, max_days: float, max_gb: float,
            days_by_label: dict[str, float] | None = None) -> tuple[int, float]:
    """Geeft (aantal verwijderde events, totale grootte in GB daarna). Per label kan een
    eigen bewaartermijn gelden (bijv. personen langer dan beweging)."""
    if not clips_dir.exists():
        return 0, 0.0
    groups = _event_groups(clips_dir)
    now = time.time()
    removed = 0
    remaining = []
    for mtime, files in groups:
        days = (days_by_label or {}).get(_label(files), max_days)
        if mtime < now - days * 86400:
            for f in files:
                f.unlink(missing_ok=True)
            removed += 1
        else:
            remaining.append((mtime, files))

    sizes = [(m, fs, sum(f.stat().st_size for f in fs if f.exists())) for m, fs in remaining]
    total = sum(s for _, _, s in sizes)
    limit = max_gb * 1024 ** 3
    for _m, files, size in sizes:  # oudste eerst
        if total <= limit:
            break
        for f in files:
            f.unlink(missing_ok=True)
        total -= size
        removed += 1

    # lege dagmappen opruimen
    for d in sorted((p for p in clips_dir.rglob("*") if p.is_dir() and ".parts" not in p.parts),
                    reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass
    return removed, total / 1024 ** 3


class RetentionWorker(threading.Thread):
    def __init__(self, clips_dir: Path, max_days: float, max_gb: float,
                 on_cleanup: Callable[[], None], interval: float = 600,
                 days_by_label: dict[str, float] | None = None):
        super().__init__(name="retention", daemon=True)
        self.clips_dir, self.max_days, self.max_gb = clips_dir, max_days, max_gb
        self.days_by_label = days_by_label or {}
        self.on_cleanup = on_cleanup
        self.interval = interval
        self._stop = threading.Event()
        self.used_gb = 0.0

    def run(self) -> None:
        while True:
            try:
                removed, self.used_gb = cleanup(self.clips_dir, self.max_days, self.max_gb,
                                                self.days_by_label)
                if removed:
                    log.info("Opruimen: %d events verwijderd, nu %.2f GB in gebruik", removed, self.used_gb)
                    self.on_cleanup()
            except Exception:  # noqa: BLE001
                log.exception("Opruimen mislukt")
            if self._stop.wait(self.interval):
                return

    def stop(self) -> None:
        self._stop.set()
