"""Load the configuration from a single YAML file.

Global settings are at the top; per camera you can override anything under `defaults`.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Labels the service knows, with the COCO classes that belong to them.
LABEL_CLASSES: dict[str, set[int]] = {
    "person": {0},                   # person
    "vehicle": {2, 3, 5, 7},         # car, motorcycle, bus, truck
    "animal": {14, 15, 16},          # bird, cat, dog
}

DEFAULTS: dict[str, Any] = {
    "pre_seconds": 10,
    "post_seconds": 30,
    "max_clip_seconds": 300,
    "record_audio": True,
    "motion": {
        "fps": 5,
        "width": 320,
        "threshold": 25,          # pixel difference (0-255) that counts as motion
        "min_area": 0.004,        # minimum share of the frame that has to move
        "cooldown_seconds": 5,    # 'motion' stays on this long after the last motion
    },
    "objects": {
        "enabled": True,
        "max_fps": "auto",        # auto = 5 with Coral/GPU, otherwise 2
        "always": False,          # also look for people without motion (CPU: once per second)
        "loiter_seconds": 30,     # unrecognised person in view this long = loitering (0 = off)
        "min_score": 0.55,
        "verify_below": 0.75,     # person/animal below this score is verified on the sharp image first (0 = off)
        "labels": {"person": True, "vehicle": True, "animal": False},
        "min_hits": 2,            # detections within the window needed for a confirmation
        "stationary_seconds": 30, # vehicle/animal in the same spot this long = stationary, no events (0 = off)
        "hit_window_seconds": 3,
    },
    "faces": {
        "enabled": False,         # face recognition for a confirmed person
        "source": "main",         # main = sharp main-stream image, sub = sub-stream
        "attempts": 8,            # at most this many attempts per event (stops at a certain name)
        "interval_seconds": 1.0,  # at least this much time between attempts
        "auto_learn": True,       # learn confidently recognised faces automatically
        "collect_unknown": True,  # collect unknown faces so they can be named later
    },
    "plates": {
        "enabled": False,         # Dutch licence plate recognition for a moving vehicle
        "attempts": 4,            # at most this many frames per event
        "interval_seconds": 1.0,
    },
    "mask": [],                   # list of polygons, points in percent [[x, y], ...]
    "zones": [],                  # [{"name": "Driveway", "points": [[x, y], ...]}]; empty = whole frame
    "hwaccel": "auto",            # "auto", "vaapi", "qsv" or "none"
}


# Settings that may change from the web page: path -> (type, min, max, description).
EDITABLE: dict[str, tuple[type, float | None, float | None, str]] = {
    "pre_seconds": (float, 3, 30, "Pre-recording (s)"),
    "post_seconds": (float, 5, 300, "Keep recording after last detection (s)"),
    "max_clip_seconds": (float, 30, 1800, "Maximum clip length (s)"),
    "motion.threshold": (int, 5, 100, "Pixel difference for motion (0-255)"),
    "motion.min_area": (float, 0.0005, 0.2, "Share of the frame that has to move"),
    "objects.min_score": (float, 0.3, 0.95, "Minimum object model confidence"),
    "objects.verify_below": (float, 0, 0.95, "Verify doubtful detections below (0 = off)"),
    "objects.labels.person": (bool, None, None, "People"),
    "objects.labels.vehicle": (bool, None, None, "Vehicles"),
    "objects.labels.animal": (bool, None, None, "Animals"),
    "objects.always": (bool, None, None, "Look for people without motion too"),
    "objects.loiter_seconds": (float, 0, 600, "Loitering after (s, 0 = off)"),
    "faces.enabled": (bool, None, None, "Face recognition"),
    "faces.auto_learn": (bool, None, None, "Learn faces automatically"),
    "faces.collect_unknown": (bool, None, None, "Collect unknown faces"),
    "plates.enabled": (bool, None, None, "Licence plate recognition (NL)"),
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
    """Validate settings from the web page; returns {path: value}."""
    if not isinstance(payload, dict):
        raise ValueError("Expected an object with settings")
    out = {}
    for path, value in payload.items():
        if path not in EDITABLE:
            raise ValueError(f"Unknown setting '{path}'")
        kind, lo, hi, label = EDITABLE[path]
        if kind is bool:
            if not isinstance(value, bool):
                raise ValueError(f"{label}: on or off")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label}: must be a number")
            if not lo <= value <= hi:
                raise ValueError(f"{label}: between {lo} and {hi}")
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
    live_stream: str | None = None  # name of the stream in go2rtc
    live_mode: str = "mse"          # go2rtc player: mse, or webrtc,mse (e.g. for H.264 Baseline on iOS)

    # quick access to frequently used settings
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
    go2rtc_url: str | None = None   # live view on the web page via go2rtc
    summary_time: str = "22:00"     # daily summary to HA (MQTT); empty = off
    ai: dict[str, Any] = field(default_factory=dict)   # descriptions by a local vision model
    auth: dict[str, Any] = field(default_factory=dict)  # passkey login: enabled, rp_id, origin, session_days
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
    """Passkey login; off by default. On requires a domain name and an https origin."""
    raw = dict(raw or {})
    out = {"enabled": bool(raw.pop("enabled", False)), "rp_id": str(raw.pop("rp_id", "")),
           "origin": str(raw.pop("origin", "")).rstrip("/"),
           "session_days": float(raw.pop("session_days", 90))}
    if raw:
        raise ValueError(f"Unknown auth setting(s): {', '.join(raw)}")
    if out["enabled"] and (not out["rp_id"] or not out["origin"].startswith("https://")
                           or out["origin"].split("//", 1)[1].split(":")[0] != out["rp_id"]):
        raise ValueError("auth: rp_id must be the domain name and origin https://<that domain>")
    return out


def _load_ai(raw: dict | None) -> dict[str, Any]:
    """`ai:` block with defaults; types and ranges are checked at startup."""
    from .ai_describe import DEFAULTS as AI_DEFAULTS
    ai = {**AI_DEFAULTS, **(raw or {})}
    unknown = set(ai) - set(AI_DEFAULTS)
    if unknown:
        raise ValueError(f"ai: unknown setting(s) {', '.join(sorted(unknown))}")
    if ai["num_gpu"] is not None and (isinstance(ai["num_gpu"], bool)
                                      or not isinstance(ai["num_gpu"], int) or ai["num_gpu"] < 1):
        raise ValueError("ai.num_gpu must be a positive integer (or left out)")
    for key, default in AI_DEFAULTS.items():
        if isinstance(default, bool):
            if not isinstance(ai[key], bool):
                raise ValueError(f"ai.{key} must be true or false")
        elif isinstance(default, (int, float)):
            if isinstance(ai[key], bool) or not isinstance(ai[key], (int, float)) or ai[key] <= 0:
                raise ValueError(f"ai.{key} must be a positive number")
    if not str(ai["base_url"]).startswith(("http://127.0.0.1:", "http://localhost:")):
        # only through the loopback tunnel: never an arbitrary or external destination
        raise ValueError("ai.base_url must be a loopback address (the SSH tunnel)")
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
        "coral_path": None,  # .tflite for the Coral; without a Coral `path` is used
        "device": "AUTO",   # AUTO = GPU first, otherwise CPU
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
            raise ValueError(f"Camera name '{name}' may only contain a-z, 0-9 and _")
        if name in seen:
            raise ValueError(f"Camera name '{name}' appears twice in the config")
        seen.add(name)
        if not cam.get("main_url") and not cam.get("onvif"):
            raise ValueError(f"Camera '{name}': provide main_url (and sub_url) or an onvif block")
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
        raise ValueError("No cameras in the config")
    return cfg
