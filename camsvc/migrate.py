"""One-time migration of data written by the earlier Dutch version (camera-service before 0.2).

Runs at startup, before the service loads anything, and only once (a marker file in the state
folder). It renames labels and fields in place:

- labels persoon/voertuig/dier/aanbellen/beweging -> person/vehicle/animal/doorbell/motion, in
  clip file names, event JSON, stationary objects, false-alarm spots, unknown faces and settings;
- event and AI fields (vals, extern, vermoedelijk, beschrijving, ...) -> their English names;
- state files (koppelcode, kleding.json, vals_<camera>.json, ...) -> their English names;
- label keys in config.yaml (a backup is kept as config.yaml.before-english).

Descriptions written by the AI in Dutch stay as they are.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

MARKER = ".migrated-english"
LABELS = {"persoon": "person", "voertuig": "vehicle", "dier": "animal", "aanbellen": "doorbell",
          "beweging": "motion"}
AI_KEYS = {"beschrijving": "description", "objecten": "objects", "kenmerken": "attributes",
           "beeldkwaliteit": "image_quality", "onzekerheden": "uncertainties", "infrarood": "infrared",
           "kleur_onbetrouwbaar": "color_unreliable", "persoon_zichtbaar": "person_visible",
           "voertuig_zichtbaar": "vehicle_visible", "dier_zichtbaar": "animal_visible",
           "duur_ms": "duration_ms", "reden": "reason", "prompt_versie": "prompt_version",
           "opgenomen": "captured", "verwerkt": "processed"}
AI_VALUES = {"image_quality": {"goed": "good", "beperkt": "limited", "onbruikbaar": "unusable"},
             "reason": {"vervangen": "replaced", "wachtrij_vol": "queue_full"}}
EXTERNAL_KEYS = {"melden": "notify", "titel": "title", "beschrijving": "description", "tijd": "time"}
STATE_FILES = {"koppelcode": "pairing_code", "setupcode": "setup_code", "kleding.json": "clothing.json",
               "draait": "running"}
_LABEL_IN_NAME = re.compile(r"^(\d{6})_(" + "|".join(LABELS) + r")(?=[_.])")


def _label(value):
    return LABELS.get(value, value)


def _path(value: str | None) -> str | None:
    """Clip-relative path with a Dutch label in the file name -> English label."""
    if not value:
        return value
    head, _, name = value.rpartition("/")
    name = _LABEL_IN_NAME.sub(lambda m: f"{m.group(1)}_{LABELS[m.group(2)]}", name)
    return f"{head}/{name}" if head else name


def migrate_event(data: dict) -> dict:
    """Event JSON of the Dutch version -> English field names and labels."""
    out = dict(data)
    if "label" in out:
        out["label"] = _label(out["label"])
    if out.get("labels"):
        out["labels"] = [_label(x) for x in out["labels"]]
    for key in ("clip", "snapshot"):
        out[key] = _path(out.get(key))
    for f in out.get("faces") or []:
        f["crop"] = _path(f.get("crop"))
    for p in out.get("plates") or []:
        p["crop"] = _path(p.get("crop"))
    if "vermoedelijk" in out:
        out["likely"] = out.pop("vermoedelijk")
    if "vals" in out:
        value = out.pop("vals")
        out["false_alarm"] = "user" if value == "gebruiker" else value
    if "extern" in out:
        ext = out.pop("extern")
        out["external"] = {EXTERNAL_KEYS.get(k, k): v for k, v in ext.items()} if ext else ext
    if out.get("ai"):
        ai = {AI_KEYS.get(k, k): v for k, v in out["ai"].items()}
        for key, values in AI_VALUES.items():
            if key in ai:
                ai[key] = values.get(ai[key], ai[key])
        out["ai"] = ai
    return out


def _rewrite_json(path: Path, fn) -> None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    new = fn(data)
    if new != data:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(new, ensure_ascii=False, indent=2))
        tmp.replace(path)


def _migrate_clips(clips_dir: Path) -> int:
    files = 0
    for f in sorted(clips_dir.rglob("*")):
        if not f.is_file() or ".parts" in f.parts:
            continue
        new_name = _path(f.name)
        target = f.with_name(new_name) if new_name != f.name else f
        if target != f and not target.exists():
            f.rename(target)
            files += 1
        if target.suffix == ".json":
            _rewrite_json(target, migrate_event)
    return files


def _migrate_state(state_dir: Path) -> None:
    for old, new in STATE_FILES.items():
        src, dst = state_dir / old, state_dir / new
        if src.exists() and not dst.exists():
            src.rename(dst)
    for src in state_dir.glob("vals_*.json"):
        dst = state_dir / src.name.replace("vals_", "false_spots_", 1)
        if not dst.exists():
            src.rename(dst)
    relabel = lambda rows: [{**r, "label": _label(r.get("label"))} for r in rows] if isinstance(rows, list) else rows  # noqa: E731
    for f in list(state_dir.glob("false_spots_*.json")) + list(state_dir.glob("stationary_*.json")):
        _rewrite_json(f, relabel)
    settings = state_dir / "settings.json"
    _rewrite_json(settings, lambda d: {cam: {re.sub(r"^objects\.labels\.(\w+)$",
                                                    lambda m: f"objects.labels.{_label(m.group(1))}", k): v
                                             for k, v in values.items()} for cam, values in d.items()})
    _rewrite_json(state_dir / "auth.json", lambda d: {**d, "credentials": [
        {{"naam": "name", "aangemaakt": "created", "laatst": "last_used"}.get(k, k): v for k, v in c.items()}
        for c in d.get("credentials", [])]})
    for f in (state_dir / "faces_unknown").glob("*.json") if (state_dir / "faces_unknown").is_dir() else []:
        _rewrite_json(f, lambda d: {**d, "crop": _path(d.get("crop"))})


def _migrate_config(config_path: Path) -> None:
    """Rename Dutch label keys in config.yaml (labels, retention per label, AI labels)."""
    try:
        text = config_path.read_text()
    except OSError:
        return
    new = text
    for nl, en in LABELS.items():
        new = re.sub(rf"(?<![\w/@:-]){nl}(?=\s*:)", en, new)          # keys: persoon: true
        new = re.sub(rf"(?<=[\[,\s]){nl}(?=\s*[,\]])", en, new)       # lists: [persoon, voertuig]
    if new != text:
        backup = config_path.with_name(config_path.name + ".before-english")
        if not backup.exists():
            backup.write_text(text)
            backup.chmod(0o600)
        config_path.write_text(new)


def migrate(config_path: Path, clips_dir: Path, state_dir: Path) -> bool:
    """Migrate once; returns True when it ran."""
    marker = state_dir / MARKER
    if marker.exists() or not state_dir.exists():
        return False
    looks_dutch = (any((state_dir / f).exists() for f in ("koppelcode", "kleding.json", "settings.json"))
                   or any(state_dir.glob("vals_*.json"))
                   or any(clips_dir.glob("*/*/*_persoon*")) or any(clips_dir.glob("*/*/*_voertuig*")))
    if looks_dutch:
        log.warning("Migrating data from the Dutch version to English names…")
        _migrate_config(config_path)
        _migrate_state(state_dir)
        n = _migrate_clips(clips_dir) if clips_dir.exists() else 0
        log.warning("Migration done (%d clip files renamed)", n)
    marker.write_text("1\n")
    return looks_dutch
