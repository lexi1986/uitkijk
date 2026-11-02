"""Configuratie laden uit één YAML-bestand.

Globale instellingen staan bovenaan; per camera kun je alles overschrijven
wat onder `defaults` staat.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Labels die de service kent, met de COCO-klassen die eronder vallen.
LABEL_CLASSES: dict[str, set[int]] = {
    "persoon": {0},                  # person
    "voertuig": {2, 3, 5, 7},        # car, motorcycle, bus, truck
    "dier": {14, 15, 16},            # bird, cat, dog
}

DEFAULTS: dict[str, Any] = {
    "pre_seconds": 10,
    "post_seconds": 30,
    "max_clip_seconds": 300,
    "record_audio": True,
    "motion": {
        "fps": 5,
        "width": 320,
        "threshold": 25,          # pixelverschil (0-255) dat als beweging telt
        "min_area": 0.004,        # minimaal deel van het beeld dat moet bewegen
        "cooldown_seconds": 5,    # zo lang blijft 'beweging' aan na de laatste beweging
    },
    "objects": {
        "enabled": True,
        "max_fps": "auto",        # auto = 5 met Coral/GPU, anders 2
        "always": False,          # ook zonder beweging naar personen kijken (CPU: 1x per seconde)
        "loiter_seconds": 30,     # persoon (niet herkend) zo lang in beeld = rondhangen (0 = uit)
        "min_score": 0.55,
        "verify_below": 0.75,     # persoon/dier onder deze score eerst nakijken op scherp beeld (0 = uit)
        "labels": {"persoon": True, "voertuig": True, "dier": False},
        "min_hits": 2,            # aantal detecties binnen het venster voor een bevestiging
        "stationary_seconds": 30, # voertuig/dier zo lang op dezelfde plek = stilstaand, geen events (0 = uit)
        "hit_window_seconds": 3,
    },
    "faces": {
        "enabled": False,         # gezichtsherkenning bij een bevestigde persoon
        "source": "main",         # main = scherp beeld uit de main-stream, sub = sub-stream
        "attempts": 8,            # maximaal zoveel pogingen per event (stopt bij een zekere naam)
        "interval_seconds": 1.0,  # minimaal zoveel tijd tussen pogingen
        "auto_learn": True,       # zeker herkende gezichten zelf bijleren
        "collect_unknown": True,  # onbekende gezichten verzamelen om later een naam te geven
    },
    "plates": {
        "enabled": False,         # NL-kentekenherkenning bij een bewegend voertuig
        "attempts": 4,            # maximaal zoveel beelden per event
        "interval_seconds": 1.0,
    },
    "mask": [],                   # lijst polygonen, punten in procenten [[x, y], ...]
    "zones": [],                  # [{"name": "Oprit", "points": [[x, y], ...]}]; leeg = hele beeld
    "hwaccel": "auto",            # "auto", "vaapi", "qsv" of "none"
}


# Instellingen die via de webpagina mogen veranderen: pad -> (type, min, max, omschrijving).
EDITABLE: dict[str, tuple[type, float | None, float | None, str]] = {
    "pre_seconds": (float, 3, 30, "Opname vooraf (s)"),
    "post_seconds": (float, 5, 300, "Doorlopen na laatste detectie (s)"),
    "max_clip_seconds": (float, 30, 1800, "Maximale cliplengte (s)"),
    "motion.threshold": (int, 5, 100, "Pixelverschil voor beweging (0-255)"),
    "motion.min_area": (float, 0.0005, 0.2, "Deel van het beeld dat moet bewegen"),
    "objects.min_score": (float, 0.3, 0.95, "Minimale zekerheid objectmodel"),
    "objects.verify_below": (float, 0, 0.95, "Twijfelgevallen nakijken onder (0 = uit)"),
    "objects.labels.persoon": (bool, None, None, "Personen"),
    "objects.labels.voertuig": (bool, None, None, "Voertuigen"),
    "objects.labels.dier": (bool, None, None, "Dieren"),
    "objects.always": (bool, None, None, "Ook zonder beweging naar personen kijken"),
    "objects.loiter_seconds": (float, 0, 600, "Rondhangen na (s, 0 = uit)"),
    "faces.enabled": (bool, None, None, "Gezichtsherkenning"),
    "faces.auto_learn": (bool, None, None, "Gezichten zelf bijleren"),
    "faces.collect_unknown": (bool, None, None, "Onbekende gezichten verzamelen"),
    "plates.enabled": (bool, None, None, "Kentekenherkenning (NL)"),
}


def get_path(settings: dict, path: str):
    for key in path.split("."):
        settings = settings[key]
    return settings


def set_path(settings: dict, path: str, value) -> None:
    *parents, last = path.split(".")
    for key in parents:
        settings = settings.setdefault(key, {})
    settings[last] = value


def validate_settings(payload: dict) -> dict:
    """Controleert instellingen uit de webpagina; geeft {pad: waarde} terug."""
    if not isinstance(payload, dict):
        raise ValueError("Verwacht een object met instellingen")
    out = {}
    for path, value in payload.items():
        if path not in EDITABLE:
            raise ValueError(f"Onbekende instelling '{path}'")
        kind, lo, hi, label = EDITABLE[path]
        if kind is bool:
            if not isinstance(value, bool):
                raise ValueError(f"{label}: aan of uit")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label}: moet een getal zijn")
            if not lo <= value <= hi:
                raise ValueError(f"{label}: tussen {lo} en {hi}")
            value = kind(value)
        out[path] = value
    return out


@dataclass
class CameraConfig:
    name: str
    friendly_name: str
    main_url: str | None
    sub_url: str | None
    onvif: dict[str, Any] | None
    settings: dict[str, Any]
    live_stream: str | None = None  # naam van de stream in go2rtc
    live_mode: str = "mse"          # go2rtc-speler: mse, of webrtc,mse (bijv. voor H.264 Baseline op iOS)

    # snelle toegang tot veelgebruikte instellingen
    @property
    def pre_seconds(self) -> float:
        return float(self.settings["pre_seconds"])

    @property
    def post_seconds(self) -> float:
        return float(self.settings["post_seconds"])

    @property
    def max_clip_seconds(self) -> float:
        return float(self.settings["max_clip_seconds"])

    @property
    def motion(self) -> dict[str, Any]:
        return self.settings["motion"]

    @property
    def objects(self) -> dict[str, Any]:
        return self.settings["objects"]

    @property
    def mask(self) -> list[list[list[float]]]:
        return self.settings["mask"] or []

    @property
    def zones(self) -> list[dict]:
        return self.settings.get("zones") or []

    def enabled_labels(self) -> list[str]:
        return [lbl for lbl, on in self.objects["labels"].items() if on and lbl in LABEL_CLASSES]


@dataclass
class Config:
    clips_dir: Path
    state_dir: Path
    retention_days: float
    retention_days_by_label: dict[str, float]
    retention_max_gb: float
    api_host: str
    api_port: int
    public_url: str
    mqtt: dict[str, Any]
    model: dict[str, Any]
    cameras: list[CameraConfig] = field(default_factory=list)
    go2rtc_url: str | None = None   # live beeld op de webpagina via go2rtc (bijv. in HA)
    summary_time: str = "22:00"     # dagoverzicht naar HA (MQTT); leeg = uit
    ai: dict[str, Any] = field(default_factory=dict)   # beschrijvingen via lokaal vision-model
    auth: dict[str, Any] = field(default_factory=dict)  # passkey-login: enabled, rp_id, origin, session_days
    log_level: str = "INFO"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


_SLUG = re.compile(r"^[a-z0-9_]+$")


def _load_auth(raw: dict | None) -> dict:
    """Passkey-login; standaard uit. Aan vraagt een domeinnaam en een https-origin."""
    raw = dict(raw or {})
    out = {"enabled": bool(raw.pop("enabled", False)), "rp_id": str(raw.pop("rp_id", "")),
           "origin": str(raw.pop("origin", "")).rstrip("/"),
           "session_days": float(raw.pop("session_days", 90))}
    if raw:
        raise ValueError(f"Onbekende auth-instelling(en): {', '.join(raw)}")
    if out["enabled"] and (not out["rp_id"] or not out["origin"].startswith("https://")
                           or out["origin"].split("//", 1)[1].split(":")[0] != out["rp_id"]):
        raise ValueError("auth: rp_id moet de domeinnaam zijn en origin https://<die domeinnaam>")
    return out


def _load_ai(raw: dict | None) -> dict[str, Any]:
    """`ai:`-blok met standaardwaarden; types en bereiken worden bij het opstarten gecontroleerd."""
    from .ai_describe import DEFAULTS as AI_DEFAULTS
    ai = {**AI_DEFAULTS, **(raw or {})}
    unknown = set(ai) - set(AI_DEFAULTS)
    if unknown:
        raise ValueError(f"ai: onbekende instelling(en) {', '.join(sorted(unknown))}")
    if ai["num_gpu"] is not None and (isinstance(ai["num_gpu"], bool)
                                      or not isinstance(ai["num_gpu"], int) or ai["num_gpu"] < 1):
        raise ValueError("ai.num_gpu moet een positief geheel getal zijn (of weglaten)")
    for key, default in AI_DEFAULTS.items():
        if isinstance(default, bool):
            if not isinstance(ai[key], bool):
                raise ValueError(f"ai.{key} moet true of false zijn")
        elif isinstance(default, (int, float)):
            if isinstance(ai[key], bool) or not isinstance(ai[key], (int, float)) or ai[key] <= 0:
                raise ValueError(f"ai.{key} moet een positief getal zijn")
    if not str(ai["base_url"]).startswith(("http://127.0.0.1:", "http://localhost:")):
        # alleen via de loopback-tunnel: nooit een willekeurige of externe bestemming
        raise ValueError("ai.base_url moet een loopback-adres zijn (de SSH-tunnel)")
    ai["labels"] = [str(x) for x in ai["labels"]]
    return ai


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}

    defaults = _deep_merge(DEFAULTS, raw.get("defaults", {}))

    storage = raw.get("storage", {})
    api = raw.get("api", {})
    mqtt = {
        "host": "localhost",
        "port": 1883,
        "username": None,
        "password": None,
        "topic_prefix": "camsvc",
        "discovery_prefix": "homeassistant",
        **raw.get("mqtt", {}),
    }
    model = {
        "path": "/opt/camera-service/models/yolo11n.onnx",
        "coral_path": None,  # .tflite voor de Coral; zonder Coral wordt `path` gebruikt
        "device": "AUTO",   # AUTO = eerst GPU, anders CPU
        "input_size": 320,
        **raw.get("model", {}),
    }

    api_port = int(api.get("port", 8080))
    cfg = Config(
        clips_dir=Path(storage.get("clips_dir", "/clips")),
        state_dir=Path(storage.get("state_dir", "/var/lib/camera-service")),
        retention_days=float(storage.get("retention_days", 7)),
        retention_days_by_label={str(k): float(v) for k, v in
                                 (storage.get("retention_days_by_label") or {}).items()},
        retention_max_gb=float(storage.get("max_gb", 25)),
        api_host=str(api.get("host", "0.0.0.0")),
        api_port=api_port,
        public_url=str(api.get("public_url", f"http://localhost:{api_port}")).rstrip("/"),
        mqtt=mqtt,
        model=model,
        log_level=str(raw.get("log_level", "INFO")).upper(),
        go2rtc_url=str((raw.get("live") or {}).get("go2rtc_url") or "").rstrip("/") or None,
        summary_time=str(raw.get("summary_time", "22:00") or ""),
        ai=_load_ai(raw.get("ai")),
        auth=_load_auth(raw.get("auth")),
    )

    seen: set[str] = set()
    for cam in raw.get("cameras", []) or []:
        name = str(cam["name"])
        if not _SLUG.match(name):
            raise ValueError(f"Cameranaam '{name}' mag alleen a-z, 0-9 en _ bevatten")
        if name in seen:
            raise ValueError(f"Cameranaam '{name}' staat dubbel in de config")
        seen.add(name)
        if not cam.get("main_url") and not cam.get("onvif"):
            raise ValueError(f"Camera '{name}': geef main_url (en sub_url) of een onvif-blok op")
        overrides = {k: v for k, v in cam.items()
                     if k not in ("name", "friendly_name", "main_url", "sub_url", "onvif", "live",
                                  "live_mode")}
        cfg.cameras.append(CameraConfig(
            name=name,
            friendly_name=str(cam.get("friendly_name", name.replace("_", " ").title())),
            main_url=cam.get("main_url"),
            sub_url=cam.get("sub_url"),
            onvif=cam.get("onvif"),
            settings=_deep_merge(defaults, overrides),
            live_stream=str(cam.get("live", name)) if cam.get("live", True) else None,
            live_mode=str(cam.get("live_mode", "mse")),
        ))

    if not cfg.cameras:
        raise ValueError("Geen camera's in de config")
    return cfg
