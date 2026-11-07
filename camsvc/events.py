"""Stored events: MP4 + JPG + JSON per event, plus the last N in memory."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

# search word -> also a match (the AI often describes the same thing differently)
SYNONYMS: dict[str, tuple[str, ...]] = {
    # the AI often calls a cargo bike with a rain cover a 'pram with a transparent hood'
    "bakfiets": ("bakfiets", "cargo bike", "box bike", "transparent hood", "transparent cover"),
    "bicycle": ("bicycle", "bike", "cargo bike"),
    "bike": ("bike", "bicycle", "cargo bike"),
    "courier": ("package", "parcel", "box", "deliver", "postman", "courier"),
    "delivery": ("package", "parcel", "box", "deliver", "postman", "courier"),
    "package": ("package", "parcel", "box"),
    "parcel": ("parcel", "package", "box"),
    "postman": ("postman", "mail", "letter", "deliver"),
    "van": ("van", "minibus", "delivery van"),
    "cat": ("cat", "kitten"),
    "dog": ("dog", "puppy"),
    "jacket": ("jacket", "coat", "parka"),
    "coat": ("coat", "jacket", "parka"),
    "cap": ("cap", "hat", "beanie"),
    "backpack": ("backpack", "rucksack"),
    "scooter": ("scooter", "moped"),
    "pram": ("pram", "stroller", "buggy", "pushchair"),
}
LAST_N = 20


@dataclass
class EventRecord:
    id: str
    camera: str
    label: str            # person / vehicle / animal / doorbell / motion
    labels: list[str]     # all labels seen during the event
    score: float | None
    start: str            # ISO time, local time zone
    end: str
    duration: float
    clip: str             # path relative to clips_dir
    snapshot: str | None
    faces: list[dict] | None = None   # [{"crop": path, "name": name or None, "score": 0-1}]
    plates: list[dict] | None = None  # [{"plate": "GKB-31-R", "name": name or None, "crop": path}]
    ai: dict | None = None            # description by the local vision model + status
    zones: list[str] | None = None    # zones the object was in (if zones are set)
    likely: list[str] | None = None   # just arrived home per HA / same clothing, face not recognised
    box: list[float] | None = None    # best detection, [x1, y1, x2, y2] as a fraction of the frame
    false_alarm: str | None = None    # probably a false alarm: "ai" or "user"
    external: dict | None = None      # verdict of the external AI in HA (comparison)
    seen: bool | None = None          # False = new (not opened yet); None = from before this feature
    objects: list[dict] | None = None   # tracked people/vehicles: label, seconds, direction

    def urls(self, public_url: str) -> dict[str, str | None]:
        return {
            "clip_url": f"{public_url}/clips/{self.clip}",
            "snapshot_url": f"{public_url}/clips/{self.snapshot}" if self.snapshot else None,
        }


class EventStore:
    def __init__(self, clips_dir: Path, public_url: str):
        self.clips_dir = clips_dir
        self.public_url = public_url
        self._recent: dict[str, deque[EventRecord]] = {}
        self._lock = threading.Lock()
        self._file_lock = threading.Lock()   # read-modify-write of event JSON
        # event JSON already read: searching over weeks (also on the NAS) reads each file once;
        # mtime check, so a change by another process (deploy/redescribe.py) is picked up too
        self._cache: dict[str, tuple[int, dict]] = {}   # path -> (mtime_ns, content)
        self.archive = None   # Archive (NAS) with older events, when configured

    def roots(self) -> list[Path]:
        """Where events live: local, plus the archive when it is reachable."""
        if self.archive is not None and self.archive.ok:
            return [self.clips_dir, self.archive.dir]
        return [self.clips_dir]

    def find(self, rel: str) -> Path | None:
        """File for a path relative to clips_dir, local or in the archive."""
        for root in self.roots():
            try:
                real = root.resolve()
                path = (real / rel).resolve()
                if real in path.parents and path.is_file():
                    # same spelling as browse() (root/camera/day/file): the cache key
                    return Path(os.path.normpath(root / rel))
            except OSError:
                self._archive_failed(root)
        return None

    def _archive_failed(self, root: Path) -> None:
        """The NAS gave an error (EIO, timeout): work local-only until the next check, so not
        every request keeps waiting on the NAS."""
        if self.archive is not None and root != self.clips_dir:
            log.warning("Archive %s gives an error; local only for now", root)
            self.archive.ok = False

    def _event_files(self, camera: str | None = None):
        """(day folder name, camera, json path) of all events, local first; an archive that
        gives an error is skipped."""
        for root in self.roots():
            try:
                cams = [camera] if camera else [d.name for d in root.iterdir()
                                                if d.is_dir() and not d.name.startswith(".")]
                found = []
                for cam in cams:
                    cam_dir = root / cam
                    if not cam_dir.is_dir():
                        continue
                    for day_dir in cam_dir.iterdir():
                        if day_dir.is_dir():
                            found += [(day_dir.name, cam, js) for js in day_dir.glob("*.json")]
            except OSError:
                self._archive_failed(root)
                continue
            yield from found

    def load_recent(self, camera: str) -> None:
        """At startup, read the last events per camera from the JSON files."""
        cam_dir = self.clips_dir / camera
        records: list[EventRecord] = []
        if cam_dir.is_dir():
            for day in sorted((d for d in cam_dir.iterdir() if d.is_dir()), reverse=True):
                for js in sorted(day.glob("*.json"), reverse=True):
                    try:
                        records.append(EventRecord(**json.loads(js.read_text())))
                    except Exception:  # noqa: BLE001
                        log.warning("Cannot read %s", js)
                    if len(records) >= LAST_N:
                        break
                if len(records) >= LAST_N:
                    break
        with self._lock:
            self._recent[camera] = deque(reversed(records), maxlen=LAST_N)

    def _write(self, js: Path, data: dict) -> None:
        """Atomic: a crash or full disk never leaves half a JSON behind."""
        tmp = js.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        tmp.replace(js)
        self._cache[str(js)] = (js.stat().st_mtime_ns, data)

    def _read(self, js: Path) -> dict:
        """Event JSON, from the cache as long as the file did not change (OSError/ValueError on error)."""
        mtime = js.stat().st_mtime_ns
        hit = self._cache.get(str(js))
        if hit is not None and hit[0] == mtime:
            return hit[1]
        data = json.loads(js.read_text())
        if len(self._cache) > 50000:     # moved events leave old keys behind
            self._cache.clear()
        self._cache[str(js)] = (mtime, data)
        return data

    def add(self, record: EventRecord) -> None:
        path = self.clips_dir / record.clip
        with self._file_lock:
            self._write(path.with_suffix(".json"), asdict(record))
        with self._lock:
            self._recent.setdefault(record.camera, deque(maxlen=LAST_N)).append(record)

    def set_ai(self, camera: str, event_id: str, ai: dict) -> None:
        with self._lock:
            for r in self._recent.get(camera, []):
                if r.id == event_id:
                    r.ai = ai

    def json_path(self, clip_rel: str) -> Path | None:
        """JSON of an event for its clip (<camera>/<date>/<time>_<label>.mp4), inside clips_dir."""
        return self.find(str(Path(clip_rel).with_suffix(".json")))

    def update(self, js: Path, fields: dict) -> dict | None:
        """Update fields of a stored event, on disk and in the recent list."""
        with self._file_lock:
            try:
                data = json.loads(js.read_text())
            except (OSError, ValueError):
                return None
            if "faces" in fields:
                # do not overwrite a name given in the meantime (player, phone) with 'unknown'
                given = {f.get("crop"): f.get("name") for f in data.get("faces") or [] if f.get("name")}
                fields = {**fields, "faces": [{**f, "name": f.get("name") or given.get(f.get("crop"))}
                                              for f in fields["faces"] or []] or None}
            data.update(fields)
            try:
                self._write(js, data)
            except OSError:
                log.exception("Cannot update %s", js)
                return None
        with self._lock:
            for r in self._recent.get(data.get("camera", ""), []):
                if r.id == data.get("id"):
                    for k, v in fields.items():
                        setattr(r, k, v)
        return data

    def set_face_name(self, crop_rel: str, name: str) -> bool:
        """Set the name for a face crop (<camera>/<date>/<time>_<label>.faceN.jpg), in the JSON
        on disk and in the recent list."""
        if ".face" not in crop_rel:
            return False
        js = self.find(crop_rel.split(".face")[0] + ".json")
        with self._file_lock:
            try:
                data = json.loads(js.read_text()) if js else None
            except (OSError, ValueError):
                data = None
            if data is None:
                return False
            for f in data.get("faces") or []:
                if f.get("crop") == crop_rel:
                    f["name"] = name
            try:
                self._write(js, data)
            except OSError:
                log.exception("Cannot update %s", js)
                return False
        with self._lock:
            for r in self._recent.get(data.get("camera", ""), []):
                for f in r.faces or []:
                    if f.get("crop") == crop_rel:
                        f["name"] = name
        return True

    def recent(self, camera: str) -> list[EventRecord]:
        """Newest first."""
        with self._lock:
            return list(reversed(self._recent.get(camera, [])))

    def forget_missing(self) -> None:
        """After cleanup: drop events whose clip is gone from the list."""
        with self._lock:
            recent = {cam: list(dq) for cam, dq in self._recent.items()}
        # outside the lock: find() may wait on a slow NAS, and add() (recording thread) waits on the lock
        gone = {id(r) for dq in recent.values() for r in dq if not self.find(r.clip)}
        with self._lock:
            for cam, dq in self._recent.items():
                self._recent[cam] = deque((r for r in dq if id(r) not in gone), maxlen=LAST_N)

    def browse(self, camera: str | None = None, label: str | None = None,
               offset: int = 0, limit: int = 60,
               person: str | None = None, plate: str | None = None, q: str | None = None,
               names: dict[str, str] | None = None, day: str | None = None,
               important: bool = False) -> tuple[list[dict], int]:
        """All events on disk, newest first, for the web page. Filters on camera and label via
        the path (<camera>/<date>/<time>_<label>[_n].json), so only the JSON files of the
        requested page need to be read."""
        seen: dict[tuple[str, str, str], Path] = {}
        # local first: while moving, an event briefly exists twice
        for day_name, cam, js in self._event_files(camera):
            if (day and day_name != day) or (label and js.stem.split("_")[1:2] != [label]):
                continue
            seen.setdefault((day_name, js.stem, cam), js)
        found: list[tuple[str, str, Path]] = [(d, stem, js) for (d, stem, _), js in seen.items()]
        found.sort(key=lambda f: (f[0], f[1]), reverse=True)
        if person:
            # people are only in the JSON: read all of them then (at most a few hundred small
            # files within the retention period)
            found = [f for f in found if self._has_person(f[2], person)]
        if plate:
            found = [f for f in found if self._has_plate(f[2], plate)]
        if important:
            found = [f for f in found if self._important(f[2], names or {})]
        if q:
            terms = [t for t in q.lower().split() if t][:6]
            found = [f for f in found if self._matches(f[2], terms, names or {})]
        page = []
        for _, _, js in found[offset:offset + limit]:
            try:
                r = EventRecord(**self._read(js))
            except Exception:  # noqa: BLE001
                log.warning("Cannot read %s", js)
                continue
            page.append(asdict(r))
        return page, len(found)

    def days(self, camera: str | None = None, label: str | None = None) -> list[dict]:
        """Number of events per day (newest first), from the file names only: fast, also on a NAS."""
        seen = {(day_name, cam, js.stem) for day_name, cam, js in self._event_files(camera)
                if not label or js.stem.split("_")[1:2] == [label]}
        counts: dict[str, int] = {}
        for day, _cam, _stem in seen:
            counts[day] = counts.get(day, 0) + 1
        return [{"day": d, "count": n} for d, n in sorted(counts.items(), reverse=True)]

    def _has_person(self, js: Path, person: str) -> bool:
        """person = a name, 'unknown' (face without a name) or 'known' (someone recognised)."""
        try:
            faces = self._read(js).get("faces") or []
        except (OSError, ValueError):
            return False
        names = [f.get("name") for f in faces]
        if person == "unknown":
            return bool(names) and not any(names)
        if person == "known":
            return any(names)
        return person in names

    def _important(self, js: Path, names: dict[str, str]) -> bool:
        """Everything except probable false alarms and your own car coming or going (vehicle
        with only known plates and no person)."""
        try:
            data = self._read(js)
        except (OSError, ValueError):
            return True
        if data.get("false_alarm"):
            return False
        labels = set(data.get("labels") or [data.get("label")])
        plates = [p.get("plate") for p in data.get("plates") or []]
        if labels <= {"vehicle"} and plates and all(p in names for p in plates):
            return False
        return True

    @staticmethod
    def search_text(data: dict, names: dict[str, str]) -> str:
        """Everything that can be searched: camera, labels, names, plates (with and without
        dashes, plus their name), zones and the AI description, objects and attributes."""
        parts = [data.get("camera", ""), " ".join(data.get("labels") or [])]
        parts += [f.get("name") or "" for f in data.get("faces") or []]
        parts += data.get("zones") or []
        for p in data.get("plates") or []:
            parts += [p.get("plate", ""), p.get("plate", "").replace("-", ""), names.get(p.get("plate"), "")]
        ai = data.get("ai") or {}
        if ai.get("status") == "ok":
            parts += [ai.get("description", ""), " ".join(ai.get("objects") or []),
                      " ".join(ai.get("attributes") or [])]
        return " ".join(parts).lower()

    def _matches(self, js: Path, terms: list[str], names: dict[str, str]) -> bool:
        try:
            text = self.search_text(self._read(js), names)
        except (OSError, ValueError):
            return False
        # every word must match, or one of its synonyms ('cargo bike' also for 'transparent hood')
        return all(any(a in text or a.replace("-", "") in text for a in SYNONYMS.get(t, (t,)))
                   for t in terms)

    def _has_plate(self, js: Path, plate: str) -> bool:
        try:
            plates = self._read(js).get("plates") or []
        except (OSError, ValueError):
            return False
        return any(p.get("plate") == plate for p in plates)

    def as_payload(self, camera: str) -> list[dict]:
        out = []
        for r in self.recent(camera):
            d = asdict(r)
            d.update(r.urls(self.public_url))
            out.append(d)
        return out


def clip_paths(clips_dir: Path, camera: str, start: datetime, label: str) -> tuple[Path, str]:
    """Returns (folder, file name without extension) for a new event."""
    day_dir = clips_dir / camera / start.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    base = f"{start.strftime('%H%M%S')}_{label}"
    n = 1
    candidate = base
    while (day_dir / f"{candidate}.mp4").exists():
        n += 1
        candidate = f"{base}_{n}"
    return day_dir, candidate
