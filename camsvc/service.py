"""Wire everything together: config, detector, cameras, MQTT, cleanup, API."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

from .camera import Camera
from .config import EDITABLE, Config, get_path, set_path
from .events import EventStore
from .mqtt import MqttBridge
from .retention import RetentionWorker

log = logging.getLogger(__name__)


def clip_duration(path: Path) -> float:
    """Duration of a (possibly interrupted, fragmented) MP4 based on its packets."""
    import av
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        first = last = None
        for pkt in c.demux(st):
            if pkt.pts is None:
                continue
            first = pkt.pts if first is None else min(first, pkt.pts)
            last = pkt.pts if last is None else max(last, pkt.pts)
        if first is None:
            return 0.0
        return float((last - first) * st.time_base)


class Service:
    def __init__(self, cfg: Config, use_mqtt: bool = True):
        self.cfg = cfg
        cfg.clips_dir.mkdir(parents=True, exist_ok=True)
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        self._switch_file = cfg.state_dir / "switches.json"
        self._switch_lock = threading.Lock()
        switches = self._load_switches()
        # settings and masks from the web page take precedence over the config
        self._settings_file = cfg.state_dir / "settings.json"
        for cam_cfg in cfg.cameras:
            for path, value in self._load_json(self._settings_file).get(cam_cfg.name, {}).items():
                if path in EDITABLE:
                    set_path(cam_cfg.settings, path, value)
        self._mask_file = cfg.state_dir / "masks.json"
        for cam_cfg in cfg.cameras:
            if cam_cfg.name in self._load_masks():
                cam_cfg.settings["mask"] = self._load_masks()[cam_cfg.name]
        self._zones_file = cfg.state_dir / "zones.json"
        for cam_cfg in cfg.cameras:
            if cam_cfg.name in self._load_json(self._zones_file):
                cam_cfg.settings["zones"] = self._load_json(self._zones_file)[cam_cfg.name]

        self.store = EventStore(cfg.clips_dir, cfg.public_url)
        self._rescue_parts()
        from .health import Health
        self.health = Health(cfg.state_dir, cfg.clips_dir)
        if self.health.previous_crash:
            log.warning("Previous run did not stop cleanly (crash or out of memory?)")
        self.auth = None
        if cfg.auth.get("enabled"):
            from .auth import Auth
            self.auth = Auth(cfg.state_dir, cfg.auth["rp_id"], cfg.auth["origin"], cfg.auth["session_days"])
        self.detector = self._load_detector()
        self.faces = self._load_faces()
        self.plates = self._load_plates()
        # without an MQTT server (no Home Assistant) everything works except the HA integration
        self.mqtt = MqttBridge(cfg, self.store) if use_mqtt and cfg.mqtt.get("host") else None
        self.ai = None
        if cfg.ai["enabled"]:
            from .ai_describe import AiDescriber
            self.ai = AiDescriber(cfg.ai, self._on_ai_result)

        self.cameras: dict[str, Camera] = {}
        for cam_cfg in cfg.cameras:
            self.cameras[cam_cfg.name] = Camera(
                cam_cfg, cfg.clips_dir, cfg.state_dir, self.detector, self.store, self.mqtt,
                switches.get(cam_cfg.name, {}), self._save_switches, self.faces, self.plates,
                self.ai)

        self.arrivals: dict[str, float] = {}   # shared with all cameras
        self.bodies = self._load_bodies()
        for cam in self.cameras.values():
            cam.arrivals = self.arrivals
            cam.bodies = self.bodies

        self.retention = RetentionWorker(cfg.clips_dir, cfg.retention_days, cfg.retention_max_gb,
                                         self._after_cleanup,
                                         days_by_label=cfg.retention_days_by_label)

    def _rescue_parts(self) -> None:
        """Keep half-written clips from a previous run (crash, power cut) as events when they
        are playable; clean up the rest."""
        from .events import EventRecord, clip_paths
        for part in (self.cfg.clips_dir / ".parts").glob("*/*.mp4.part"):
            event_id = part.name.split(".")[0]
            meta_file = part.parent / f"{event_id}.meta.json"
            snap_file = part.parent / f"{event_id}.jpg"
            try:
                meta = json.loads(meta_file.read_text())
                duration = clip_duration(part)
                if duration < 1.0:
                    raise ValueError("too short")
                start = datetime.fromisoformat(meta["start"])
                day_dir, base = clip_paths(self.cfg.clips_dir, meta["camera"], start, meta["label"])
                mp4 = day_dir / f"{base}.mp4"
                part.replace(mp4)
                snap = None
                if snap_file.exists():
                    snap_file.replace(day_dir / f"{base}.jpg")
                    snap = str((day_dir / f"{base}.jpg").relative_to(self.cfg.clips_dir))
                self.store.add(EventRecord(
                    id=meta["id"], camera=meta["camera"], label=meta["label"], labels=[meta["label"]],
                    score=None, start=meta["start"], end=meta["start"], duration=round(duration, 1),
                    clip=str(mp4.relative_to(self.cfg.clips_dir)), snapshot=snap))
                log.warning("Clip recovered after interruption: %s (%.0f s)", mp4.name, duration)
            except Exception as exc:  # noqa: BLE001 - recovery must never block startup
                log.info("Partial clip %s not recoverable (%s); removed", part.name, exc)
                part.unlink(missing_ok=True)
            meta_file.unlink(missing_ok=True)
            snap_file.unlink(missing_ok=True)
        for stale in (self.cfg.clips_dir / ".parts").glob("*/*"):
            if stale.suffix in (".json", ".jpg") and not stale.with_name(
                    stale.name.split(".")[0] + ".mp4.part").exists():
                stale.unlink(missing_ok=True)

    def _load_detector(self):
        needed = any(c.objects["enabled"] and c.enabled_labels() for c in self.cfg.cameras)
        if not needed:
            log.info("Object detection is off for all cameras; no model loaded")
            return None
        from .detector import ObjectDetector
        size = int(self.cfg.model["input_size"])
        coral = self.cfg.model.get("coral_path")
        if coral and Path(coral).exists():
            try:
                return ObjectDetector(coral, self.cfg.model["device"], size)
            except Exception as exc:  # noqa: BLE001
                log.warning("Coral not usable (%s); falling back to %s", exc, self.cfg.model["path"])
        elif coral:
            log.warning("Coral model %s not found; see the guide 'Google Coral'", coral)
        path = Path(self.cfg.model["path"])
        if not path.exists():
            log.error("Model %s not found; object detection off, motion detection only. "
                      "See the guide: 'Exporting the model'.", path)
            return None
        try:
            return ObjectDetector(path, self.cfg.model["device"], size)
        except Exception as exc:  # noqa: BLE001
            log.error("Loading the object model failed (%s); motion detection only", exc)
            return None

    def _load_plates(self, force: bool = False):
        if not force and not any(c.settings["plates"]["enabled"] for c in self.cfg.cameras):
            return None
        try:
            import fast_alpr  # noqa: F401 - only check that the package is installed
            from .plates import PlateReader
            reader = PlateReader(self.cfg.state_dir)
        except ImportError as exc:
            log.error("Plate recognition not available (%s); pip install -r requirements.txt", exc)
            return None
        threading.Thread(target=reader.warm_up, name="plates-load", daemon=True).start()
        return reader

    def _load_bodies(self):
        """Recognition by clothing/build; only useful together with face recognition."""
        if self.faces is None:
            return None
        from .reid import BodyMatcher
        models_dir = Path(self.cfg.model["path"]).parent
        if not BodyMatcher.available(models_dir):
            log.info("No clothing/build model in %s; skipped", models_dir)
            return None
        try:
            matcher = BodyMatcher(models_dir, self.cfg.state_dir / "clothing.json")
            log.info("Recognition by clothing/build enabled")
            return matcher
        except Exception as exc:  # noqa: BLE001
            log.error("Loading the clothing/build model failed (%s)", exc)
            return None

    def _load_faces(self, force: bool = False):
        if not force and not any(c.settings["faces"]["enabled"] for c in self.cfg.cameras):
            return None
        from .faces import FaceRecognizer
        models_dir = Path(self.cfg.model["path"]).parent
        if not FaceRecognizer.available(models_dir):
            log.error("Face models missing in %s; face recognition off. "
                      "See the guide: 'Face recognition'.", models_dir)
            return None
        try:
            return FaceRecognizer(models_dir, self.cfg.state_dir / "faces")
        except Exception as exc:  # noqa: BLE001
            log.error("Loading face recognition failed (%s)", exc)
            return None

    # --- settings from the web page -----------------------------------------
    @staticmethod
    def _load_json(path: Path) -> dict:
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, ValueError):
            return {}

    def _save_json(self, path: Path, data: dict) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)

    def camera_settings(self, cam: str) -> dict:
        settings = self.cameras[cam].cfg.settings
        return {path: get_path(settings, path) for path in EDITABLE}

    def set_settings(self, cam: str, values: dict) -> dict:
        if values.get("faces.enabled") and self.faces is None:
            self.faces = self._load_faces(force=True)
            if self.faces is None:
                raise ValueError("Face models are missing on the server")
        if values.get("plates.enabled") and self.plates is None:
            self.plates = self._load_plates(force=True)
            if self.plates is None:
                raise ValueError("Plate recognition is not installed on the server")
        self.cameras[cam].apply_settings(values, self.faces, self.plates)
        with self._switch_lock:
            data = self._load_json(self._settings_file)
            data.setdefault(cam, {}).update(values)
            self._save_json(self._settings_file, data)
        if self.mqtt and ("faces.enabled" in values or "plates.enabled" in values):
            self.mqtt.reannounce()      # add or remove the 'Face' sensor
        return self.camera_settings(cam)

    # --- masks from the web page --------------------------------------------
    def _load_masks(self) -> dict:
        try:
            return json.loads(self._mask_file.read_text())
        except (FileNotFoundError, ValueError):
            return {}

    def set_mask(self, cam: str, polygons: list) -> None:
        self.cameras[cam].set_mask(polygons)
        with self._switch_lock:
            masks = self._load_masks()
            masks[cam] = polygons
            tmp = self._mask_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(masks, indent=2))
            tmp.replace(self._mask_file)

    def set_zones(self, cam: str, zones: list) -> None:
        self.cameras[cam].set_zones(zones)
        with self._switch_lock:
            data = self._load_json(self._zones_file)
            data[cam] = zones
            self._save_json(self._zones_file, data)

    # --- keep switches across restarts --------------------------------------
    def _load_switches(self) -> dict:
        try:
            return json.loads(self._switch_file.read_text())
        except (FileNotFoundError, ValueError):
            return {}

    def _save_switches(self) -> None:
        with self._switch_lock:
            data = {n: {"detect": c.detect_enabled, "record": c.record_enabled}
                    for n, c in getattr(self, "cameras", {}).items()}
            tmp = self._switch_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(self._switch_file)

    def _on_switch(self, cam: str, which: str, on: bool) -> None:
        camera = self.cameras.get(cam)
        if camera:
            camera.set_switch(which, on)

    def _on_ai_result(self, cam: str, event_id: str, result: dict) -> None:
        camera = self.cameras.get(cam)
        if camera:
            camera.attach_ai(event_id, result)

    def _on_trigger(self, cam: str, label: str) -> None:
        camera = self.cameras.get(cam)
        if camera and label in ("doorbell", "motion"):
            camera.manual_trigger(label)

    def _state(self, cam: str) -> dict[str, bool]:
        c = self.cameras[cam]
        return {"detect": c.detect_enabled, "record": c.record_enabled, "online": c.online,
                "motion": c.motion_active, **c.object_active, "loitering": c.loitering,
                "face": c.face_name, "plate": c.plate}

    def _after_cleanup(self) -> None:
        self.store.forget_missing()
        if self.mqtt:
            for name in self.cameras:
                self.mqtt.publish_recent(name)

    # --- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        for cam in self.cameras.values():
            cam.start()
        if self.mqtt:
            self.mqtt.face_handler = self._on_face_answer
            self.mqtt.presence_handler = self._on_presence
            self.mqtt.external_handler = self._on_external_ai
            self.mqtt.start([c.cfg for c in self.cameras.values()], self._on_switch, self._state,
                            self._on_trigger)
        self.retention.start()
        self._stop_evt = threading.Event()
        if self.mqtt and self.cfg.summary_time:
            threading.Thread(target=self._summary_loop, name="daily-summary", daemon=True).start()
        if self.mqtt:
            threading.Thread(target=self._health_loop, name="health", daemon=True).start()
        if self.ai:
            self.ai.start()
            log.info("AI descriptions enabled (%s via %s)", self.cfg.ai["model"], self.cfg.ai["base_url"])
        log.info("Uitkijk started with %d camera(s)", len(self.cameras))

    def health_snapshot(self) -> dict:
        return self.health.snapshot(self.ai.state if self.ai else "off",
                                    sum(c.online for c in self.cameras.values()), len(self.cameras))

    def _health_loop(self) -> None:
        """Every minute CPU, memory, disk and AI status to HA; after a crash one alert."""
        self.mqtt.connected.wait(120)
        if self.health.previous_crash:
            self.mqtt.publish_server_event("restart_after_crash", {
                "started": self.health.started.isoformat(timespec="seconds")})
        while True:
            try:
                self.mqtt.publish_health(self.health_snapshot())
            except Exception:  # noqa: BLE001
                log.exception("Publishing health failed")
            if self._stop_evt.wait(60):
                return

    def _on_external_ai(self, cam: str, result: dict) -> None:
        """Store the verdict of the external AI (HA script camera_ai_check) with the event."""
        camera = self.cameras.get(cam)
        if camera is None:
            return
        result["time"] = datetime.now().astimezone().isoformat(timespec="seconds")
        if not camera.attach_external(result):
            log.info("[%s] external AI verdict without a matching event", cam)

    def mark_false(self, clip_rel: str, false: bool) -> dict:
        """Mark an event as a false alarm (or undo that). For an object with a position the camera
        remembers that spot, so the same thing there does not trigger an event again."""
        js = self.store.json_path(clip_rel)
        if js is None:
            raise KeyError("Event not found")
        data = json.loads(js.read_text())
        camera = self.cameras.get(data.get("camera", ""))
        learned = False
        if false:
            if camera and data.get("box") and data.get("label") in camera.object_active:
                learned = camera.add_false_spot(data["label"], data["box"], data["id"])
            data = self.store.update(js, {"false_alarm": "user"})
        else:
            if camera:
                camera.remove_false_spot(data["id"])
            data = self.store.update(js, {"false_alarm": None})
        return {"false_alarm": data.get("false_alarm") if data else None, "spot_learned": learned}

    def _on_presence(self, name: str, home: bool) -> None:
        """HA reports that a resident arrives home (or leaves)."""
        if home:
            self.arrivals[name] = time.monotonic()
            log.info("%s arrived home (according to HA)", name)
        else:
            self.arrivals.pop(name, None)

    def _on_face_answer(self, sample: str, name: str | None, ignore: bool) -> None:
        """Answer to 'Who is this?' from a notification: name the whole group or ignore it."""
        if self.faces is None:
            return
        group = self.faces.unknown.group_of(sample)
        if group is None:
            log.info("Face %s already handled or expired", sample)
            return
        try:
            if ignore or not name:
                self.faces.unknown.dismiss(group["ids"])
                log.info("Unknown face ignored from notification (%d photos)", group["count"])
                return
            updates = self.faces.unknown.assign(group["ids"], name)
            for crop, who in updates:
                self.store.set_face_name(crop, who)
            log.info("Face learned from notification as %s (%d events updated)", name, len(updates))
        except (ValueError, KeyError) as exc:
            log.warning("Face from notification not processed: %s", exc)

    def _summary_loop(self) -> None:
        """Send the daily summary to HA every day at summary_time."""
        from .summary import summarize_day
        sent = None
        today_at = 0.0
        while not self._stop_evt.wait(20):
            now = datetime.now().astimezone()
            if time.monotonic() - today_at >= 300:      # running total for the HA dashboard
                today_at = time.monotonic()
                try:
                    self.mqtt.publish_today(summarize_day(
                        self.cfg.clips_dir, now.date(), self.plates.known if self.plates else {}))
                except Exception:  # noqa: BLE001
                    log.exception("Building today's running total failed")
            if now.strftime("%H:%M") != self.cfg.summary_time or sent == now.date():
                continue
            sent = now.date()
            try:
                s = summarize_day(self.cfg.clips_dir, now.date(), self.plates.known if self.plates else {})
                self.mqtt.publish_summary(s)
                log.info("Daily summary sent: %s", s["text"])
            except Exception:  # noqa: BLE001
                log.exception("Building the daily summary failed")

    def stop(self) -> None:
        log.info("Stopping…")
        if hasattr(self, "_stop_evt"):
            self._stop_evt.set()
        for cam in self.cameras.values():
            cam.stop()
        self.retention.stop()
        if self.ai:
            self.ai.stop()
        deadline = time.monotonic() + 20      # together within systemd's TimeoutStopSec
        for cam in self.cameras.values():
            cam.recorder.join(timeout=max(0.5, deadline - time.monotonic()))   # finish the clip
        if self.mqtt:
            self.mqtt.stop()
        self.health.clean_stop()
