"""Dagoverzicht: alle events van één dag samengevat, voor de webpagina en de avondmelding."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

PACKAGE_RE = re.compile(r"\b(pakket|pakje)", re.IGNORECASE)


def _time(iso: str) -> str:
    return iso[11:16]


def _known_plate(plate: str, names: dict[str, str]) -> str | None:
    """Kenteken bij de bekende lijst zetten, ook met één verkeerd gelezen teken."""
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
    """Tellingen per soort en camera, wie er langs kwam, auto's, pakketjes en een korte tekst.
    Waarschijnlijk valse events (AI of gebruiker) tellen niet mee, alleen hun aantal."""
    plate_names = plate_names or {}
    events = _day_events(clips_dir, day)
    false_count = sum(1 for e in events if e.get("vals"))
    events = [e for e in events if not e.get("vals")]

    per_label = Counter(e.get("label") for e in events)
    per_camera = Counter(e.get("camera") for e in events)
    people: dict[str, dict] = {}
    unknown = 0         # events met een gezicht zonder naam
    person_events = 0
    cars: dict[str, dict] = {}
    packages = 0
    hours = Counter()
    for e in events:
        start = e.get("start", "")
        hours[start[11:13]] += 1
        names = sorted({f["name"] for f in e.get("faces") or [] if f.get("name")}
                       or set(e.get("vermoedelijk") or []))   # net thuisgekomen volgens HA
        for n in names:
            p = people.setdefault(n, {"aantal": 0, "eerst": _time(start), "laatst": _time(start)})
            p["aantal"] += 1
            p["laatst"] = _time(start)
        if "persoon" in (e.get("labels") or []):
            person_events += 1
        if e.get("faces") and not names:
            unknown += 1
        for pl in e.get("plates") or []:
            plate = _known_plate(pl["plate"], plate_names) or pl["plate"]
            c = cars.setdefault(plate, {"plate": plate, "naam": plate_names.get(plate),
                                        "aantal": 0, "eerst": _time(start)})
            c["aantal"] += 1
        ai = e.get("ai") or {}
        if ai.get("status") == "ok" and PACKAGE_RE.search(" ".join((ai.get("kenmerken") or [])
                                                                  + (ai.get("objecten") or []))):
            packages += 1

    busiest = max(hours.items(), key=lambda kv: kv[1])[0] + ":00" if hours else None
    out = {
        "datum": day.isoformat(),
        "events": len(events),
        "per_soort": dict(per_label),
        "per_camera": dict(per_camera),
        "personen": people,
        "persoon_events": person_events,
        "onbekende_personen": unknown,
        "aanbellen": per_label.get("aanbellen", 0),
        "pakketten": packages,
        "autos": sorted(cars.values(), key=lambda c: c["eerst"]),
        "drukste_uur": busiest,
        "vals": false_count,
    }
    out["tekst"] = summary_text(out)
    return out


def summary_text(s: dict) -> str:
    """Eén korte alinea, geschikt voor een melding."""
    parts = []
    if s["aanbellen"]:
        parts.append(f"{s['aanbellen']}× aangebeld")
    if s["pakketten"]:
        parts.append(f"{s['pakketten']} pakket{'je' if s['pakketten'] == 1 else 'jes'}")
    who = [f"{n} ({p['aantal']}×, eerst {p['eerst']})" for n, p in
           sorted(s["personen"].items(), key=lambda kv: kv[1]["eerst"])]
    if s["onbekende_personen"]:
        who.append(f"{s['onbekende_personen']}× onbekend gezicht")
    if s["persoon_events"]:
        parts.append(f"{s['persoon_events']}× een persoon" + (": " + ", ".join(who) if who else ""))
    if s["autos"]:
        parts.append("Auto's: " + ", ".join(c["naam"] or f"onbekend {c['plate']}" for c in s["autos"]))
    weggelaten = f"{s['vals']}× niets aan de hand weggelaten" if s.get("vals") else ""
    if not parts:
        text = (f"Rustige dag: {s['events']} opnames, geen bezoek of auto's." if s["events"]
                else "Niets opgenomen vandaag.")
        return f"{text} {weggelaten}." if weggelaten else text
    if weggelaten:
        parts.append(weggelaten)
    return f"{s['events']} opnames. " + ". ".join(parts) + "."


def _local_verdict(e: dict) -> bool | None:
    """Zou de lokale AI dit melden? Ja als die zag waarvoor het event startte. None als er geen
    (nieuw genoeg) lokaal oordeel is."""
    from .ai_describe import false_alarm
    ai = e.get("ai") or {}
    if ai.get("status") != "ok" or "persoon_zichtbaar" not in ai:
        return None
    return not false_alarm(e.get("labels") or [e.get("label")], ai)


def compare_ai(clips_dir: Path, until: date, days: int = 7) -> dict:
    """Vergelijkingsweek: per event het oordeel van de externe AI (melden ja/nee, in HA) naast
    dat van de lokale AI (zag die de persoon/het object?). Laat zien waar ze het oneens zijn."""
    rows = []
    for i in range(days):
        for e in _day_events(clips_dir, until - timedelta(days=i)):
            ext = e.get("extern")
            local = _local_verdict(e)
            if ext is None or local is None:
                continue
            ai = e["ai"]
            rows.append({"start": e["start"], "camera": e["camera"], "label": e["label"],
                         "clip": e["clip"], "snapshot": e.get("snapshot"),
                         "extern": bool(ext.get("melden")), "lokaal": local,
                         "vals": e.get("vals"),
                         "extern_tekst": ext.get("titel") or ext.get("beschrijving", ""),
                         "lokaal_tekst": ai.get("beschrijving", ""),
                         "lokaal_ms": ai.get("duur_ms")})
    rows.sort(key=lambda r: r["start"], reverse=True)
    agree = sum(r["extern"] == r["lokaal"] for r in rows)
    ms = sorted(r["lokaal_ms"] for r in rows if r["lokaal_ms"])
    return {
        "van": (until - timedelta(days=days - 1)).isoformat(), "tot": until.isoformat(),
        "events": len(rows),
        "eens": agree,
        "alleen_extern": sum(r["extern"] and not r["lokaal"] for r in rows),
        "alleen_lokaal": sum(r["lokaal"] and not r["extern"] for r in rows),
        "lokaal_mediaan_ms": ms[len(ms) // 2] if ms else None,
        "oneens": [r for r in rows if r["extern"] != r["lokaal"]][:50],
    }


def parse_day(value: str | None) -> date:
    if not value:
        return datetime.now().astimezone().date()
    return date.fromisoformat(value)
