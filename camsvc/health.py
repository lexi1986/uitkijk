"""Gezondheid van de server zelf: CPU, geheugen, schijf en of de vorige keer netjes stopte.

Geheugen komt uit de cgroup van de service (systemd: MemoryMax), zodat het percentage zegt hoe
dicht we bij een OOM-kill zitten. Buiten systemd valt het terug op het RSS van het proces.
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime
from pathlib import Path

CRASH_MARKER = "draait"     # bestaat zolang de service draait; blijft staan na een crash/OOM-kill


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _cgroup_dir() -> Path | None:
    line = _read(Path("/proc/self/cgroup"))
    if not line:
        return None
    for row in line.splitlines():
        if row.startswith("0::"):
            d = Path("/sys/fs/cgroup") / row[3:].lstrip("/")
            return d if (d / "memory.current").exists() else None
    return None


class Health:
    def __init__(self, state_dir: Path, clips_dir: Path):
        self.clips_dir = clips_dir
        self.started = datetime.now().astimezone()
        self._marker = state_dir / CRASH_MARKER
        # bestond de marker al, dan is de vorige keer niet netjes gestopt
        self.previous_crash = self._marker.exists()
        self._marker.write_text(self.started.isoformat(timespec="seconds"))
        self._cgroup = _cgroup_dir()
        self._cpu_last = (time.monotonic(), self._cpu_seconds())

    def clean_stop(self) -> None:
        self._marker.unlink(missing_ok=True)

    @staticmethod
    def _cpu_seconds() -> float:
        t = os.times()
        return t.user + t.system + t.children_user + t.children_system

    def _cpu_percent(self) -> float:
        now, cpu = time.monotonic(), self._cpu_seconds()
        then, cpu_then = self._cpu_last
        self._cpu_last = (now, cpu)
        if now - then <= 0:
            return 0.0
        return round(100 * (cpu - cpu_then) / (now - then) / (os.cpu_count() or 1), 1)

    def _memory(self) -> tuple[float, float | None]:
        """(MB in gebruik, % van de limiet of None)."""
        if self._cgroup is not None:
            cur = _read(self._cgroup / "memory.current")
            limit = _read(self._cgroup / "memory.max")
            if cur and cur.isdigit():
                used = int(cur)
                if limit and limit.isdigit():
                    return round(used / 1e6), round(100 * used / int(limit), 1)
                total = self._meminfo_total()
                return round(used / 1e6), round(100 * used / total, 1) if total else None
        status = _read(Path("/proc/self/status")) or ""
        rss = next((int(r.split()[1]) * 1024 for r in status.splitlines() if r.startswith("VmRSS:")), 0)
        total = self._meminfo_total()
        return round(rss / 1e6), round(100 * rss / total, 1) if total else None

    @staticmethod
    def _meminfo_total() -> int | None:
        info = _read(Path("/proc/meminfo")) or ""
        kb = next((int(r.split()[1]) for r in info.splitlines() if r.startswith("MemTotal:")), None)
        return kb * 1024 if kb else None

    def snapshot(self, ai_state: str, cameras_online: int, cameras_total: int) -> dict:
        mb, pct = self._memory()
        disk = shutil.disk_usage(self.clips_dir)
        return {
            "gestart": self.started.isoformat(timespec="seconds"),
            "cpu": self._cpu_percent(),
            "geheugen": pct,
            "geheugen_mb": mb,
            "schijf": round(100 * disk.used / disk.total, 1),
            "schijf_vrij_gb": round(disk.free / 1e9, 1),
            "ai_server": ai_state,
            "cameras_online": cameras_online,
            "cameras_totaal": cameras_total,
        }
