"""Daily summary: all events of one day summarised, for the web page and the evening notification."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

PACKAGE_RE = re.compile(r"\b(package|parcel)", re.IGNORECASE)


def _time(iso: str) -> str:
    return iso[11:16]


def _known_plate(plate: str, names: dict[str, str]) -> str | None:
    """Match a plate to the known list, also with one misread character."""
    if plate in names:
        return plate
    flat = plate.replace("-", "")
    for known in names:
        k = known.replace("-", "")
        if len(k) == len(flat) and sum(a != b for a, b in zip(k, flat)) <= 1:
            return known
    return None


def _day_events(clips_dir: Path, day: date) -> list[dict]:
    events = []
    for cam_dir in sorted(d for d in clips_dir.iterdir() if d.is_dir() and not d.name.startswith(".")):
        for js in sorted((cam_dir / day.isoformat()).glob("*.json")):
            try:
                events.append(json.loads(js.read_text()))
            except (OSError, ValueError):
                continue
    return sorted(events, key=lambda e: e.get("start", ""))


def summarize_day(clips_dir: Path, day: date, plate_names: dict[str, str] | None = None) -> dict:
    """Counts per label and camera, who came by, vehicles, packages and a short text.
    Probable false alarms (AI or user) are left out; only their number is reported."""
    plate_names = plate_names or {}
    events = _day_events(clips_dir, day)
    false_count = sum(1 for e in events if e.get("false_alarm"))
    events = [e for e in events if not e.get("false_alarm")]

    per_label = Counter(e.get("label") for e in events)
    per_camera = Counter(e.get("camera") for e in events)
    people: dict[str, dict] = {}
    unknown = 0         # events with a face without a name
    person_events = 0
    vehicles: dict[str, dict] = {}
    packages = 0
    hours = Counter()
    for e in events:
        start = e.get("start", "")
        hours[start[11:13]] += 1
        names = sorted({f["name"] for f in e.get("faces") or [] if f.get("name")}
                       or set(e.get("likely") or []))   # just arrived home according to HA
        for n in names:
            p = people.setdefault(n, {"count": 0, "first": _time(start), "last": _time(start)})
            p["count"] += 1
            p["last"] = _time(start)
        if "person" in (e.get("labels") or []):
            person_events += 1
        if e.get("faces") and not names:
            unknown += 1
        for pl in e.get("plates") or []:
            plate = _known_plate(pl["plate"], plate_names) or pl["plate"]
            v = vehicles.setdefault(plate, {"plate": plate, "name": plate_names.get(plate),
                                            "count": 0, "first": _time(start)})
            v["count"] += 1
        ai = e.get("ai") or {}
        if ai.get("status") == "ok" and PACKAGE_RE.search(" ".join((ai.get("attributes") or [])
                                                                  + (ai.get("objects") or []))):
            packages += 1

    busiest = max(hours.items(), key=lambda kv: kv[1])[0] + ":00" if hours else None
    out = {
        "date": day.isoformat(),
        "events": len(events),
        "per_label": dict(per_label),
        "per_camera": dict(per_camera),
        "people": people,
        "person_events": person_events,
        "unknown_faces": unknown,
        "doorbell": per_label.get("doorbell", 0),
        "packages": packages,
        "vehicles": sorted(vehicles.values(), key=lambda v: v["first"]),
        "busiest_hour": busiest,
        "false_alarms": false_count,
    }
    out["text"] = summary_text(out)
    return out


def summary_text(s: dict) -> str:
    """One short paragraph, suitable for a notification."""
    parts = []
    if s["doorbell"]:
        parts.append(f"Doorbell rang {s['doorbell']}×")
    if s["packages"]:
        parts.append(f"{s['packages']} package{'' if s['packages'] == 1 else 's'}")
    who = [f"{n} ({p['count']}×, first {p['first']})" for n, p in
           sorted(s["people"].items(), key=lambda kv: kv[1]["first"])]
    if s["unknown_faces"]:
        who.append(f"{s['unknown_faces']}× unknown face")
    if s["person_events"]:
        parts.append(f"{s['person_events']}× a person" + (": " + ", ".join(who) if who else ""))
    if s["vehicles"]:
        parts.append("Vehicles: " + ", ".join(v["name"] or f"unknown {v['plate']}" for v in s["vehicles"]))
    left_out = f"{s['false_alarms']}× nothing to worry about left out" if s.get("false_alarms") else ""
    if not parts:
        text = (f"Quiet day: {s['events']} recordings, no visitors or vehicles." if s["events"]
                else "Nothing recorded today.")
        return f"{text} {left_out}." if left_out else text
    if left_out:
        parts.append(left_out)
    return f"{s['events']} recordings. " + ". ".join(parts) + "."


def _local_verdict(e: dict) -> bool | None:
    """Would the local AI notify about this? Yes when it saw what triggered the event. None when
    there is no (recent enough) local verdict."""
    from .ai_describe import false_alarm
    ai = e.get("ai") or {}
    if ai.get("status") != "ok" or "person_visible" not in ai:
        return None
    return not false_alarm(e.get("labels") or [e.get("label")], ai)


def compare_ai(clips_dir: Path, until: date, days: int = 7) -> dict:
    """Comparison week: per event the verdict of the external AI (notify yes/no, in HA) next to
    that of the local AI (did it see the person/object?). Shows where they disagree."""
    rows = []
    for i in range(days):
        for e in _day_events(clips_dir, until - timedelta(days=i)):
            ext = e.get("external")
            local = _local_verdict(e)
            if ext is None or local is None:
                continue
            ai = e["ai"]
            rows.append({"start": e["start"], "camera": e["camera"], "label": e["label"],
                         "clip": e["clip"], "snapshot": e.get("snapshot"),
                         "external": bool(ext.get("notify")), "local": local,
                         "false_alarm": e.get("false_alarm"),
                         "external_text": ext.get("title") or ext.get("description", ""),
                         "local_text": ai.get("description", ""),
                         "local_ms": ai.get("duration_ms")})
    rows.sort(key=lambda r: r["start"], reverse=True)
    agree = sum(r["external"] == r["local"] for r in rows)
    ms = sorted(r["local_ms"] for r in rows if r["local_ms"])
    return {
        "from": (until - timedelta(days=days - 1)).isoformat(), "until": until.isoformat(),
        "events": len(rows),
        "agree": agree,
        "only_external": sum(r["external"] and not r["local"] for r in rows),
        "only_local": sum(r["local"] and not r["external"] for r in rows),
        "local_median_ms": ms[len(ms) // 2] if ms else None,
        "disagree": [r for r in rows if r["external"] != r["local"]][:50],
    }


def parse_day(value: str | None) -> date:
    if not value:
        return datetime.now().astimezone().date()
    return date.fromisoformat(value)
