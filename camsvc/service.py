"""Alles aan elkaar knopen: config, detector, camera's, MQTT, opruimen, API."""

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
    """Duur van een (mogelijk afgebroken, gefragmenteerde) MP4 aan de hand van de pakketten."""
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
        # instellingen en maskers uit de webpagina gaan voor op die uit de config
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
            log.warning("Vorige keer niet netjes gestopt (crash of geheugentekort?)")
        self.auth = None
        if cfg.auth.get("enabled"):
            from .auth import Auth
            self.auth = Auth(cfg.state_dir, cfg.auth["rp_id"], cfg.auth["origin"], cfg.auth["session_days"])
        self.detector = self._load_detector()
        self.faces = self._load_faces()
        self.plates = self._load_plates()
        # zonder MQTT-server (geen Home Assistant) werkt alles behalve de koppeling met HA
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

        self.arrivals: dict[str, float] = {}   # gedeeld met alle camera's
        self.bodies = self._load_bodies()
        for cam in self.cameras.values():
            cam.arrivals = self.arrivals
            cam.bodies = self.bodies

        self.retention = RetentionWorker(cfg.clips_dir, cfg.retention_days, cfg.retention_max_gb,
                                         self._after_cleanup,
                                         days_by_label=cfg.retention_days_by_label)

    def _rescue_parts(self) -> None:
        """Halve clips van een vorige run (crash, stroomuitval) als event bewaren als ze
        afspeelbaar zijn; de rest opruimen."""
        from .events import EventRecord, clip_paths
        for part in (self.cfg.clips_dir / ".parts").glob("*/*.mp4.part"):
            event_id = part.name.split(".")[0]
            meta_file = part.parent / f"{event_id}.meta.json"
            snap_file = part.parent / f"{event_id}.jpg"
            try:
                meta = json.loads(meta_file.read_text())
                duration = clip_duration(part)
                if duration < 1.0:
                    raise ValueError("te kort")
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
                log.warning("Clip na onderbreking gered: %s (%.0f s)", mp4.name, duration)
            except Exception as exc:  # noqa: BLE001 - herstel mag het opstarten nooit blokkeren
                log.info("Halve clip %s niet te redden (%s); opgeruimd", part.name, exc)
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
            log.info("Objectdetectie staat bij alle camera's uit; geen model geladen")
            return None
        from .detector import ObjectDetector
        size = int(self.cfg.model["input_size"])
        coral = self.cfg.model.get("coral_path")
        if coral and Path(coral).exists():
            try:
                return ObjectDetector(coral, self.cfg.model["device"], size)
            except Exception as exc:  # noqa: BLE001
                log.warning("Coral niet bruikbaar (%s); terug naar %s", exc, self.cfg.model["path"])
        elif coral:
            log.warning("Coral-model %s niet gevonden; zie README 'Google Coral'", coral)
        path = Path(self.cfg.model["path"])
        if not path.exists():
            log.error("Model %s niet gevonden; objectdetectie uit, alleen bewegingsdetectie. "
                      "Zie README: 'Model exporteren'.", path)
            return None
        try:
            return ObjectDetector(path, self.cfg.model["device"], size)
        except Exception as exc:  # noqa: BLE001
            log.error("Objectmodel laden mislukt (%s); alleen bewegingsdetectie", exc)
            return None

    def _load_plates(self, force: bool = False):
        if not force and not any(c.settings["plates"]["enabled"] for c in self.cfg.cameras):
            return None
        try:
            import fast_alpr  # noqa: F401 - alleen controleren of het pakket er is
            from .plates import PlateReader
            reader = PlateReader(self.cfg.state_dir)
        except ImportError as exc:
            log.error("Kentekenherkenning niet beschikbaar (%s); pip install -r requirements.txt", exc)
            return None
        threading.Thread(target=reader.warm_up, name="plates-load", daemon=True).start()
        return reader

    def _load_bodies(self):
        """Herkennen aan kleding/postuur; alleen zinvol samen met gezichtsherkenning."""
        if self.faces is None:
            return None
        from .reid import BodyMatcher
        models_dir = Path(self.cfg.model["path"]).parent
        if not BodyMatcher.available(models_dir):
            log.info("Geen model voor kleding/postuur in %s; overgeslagen", models_dir)
            return None
        try:
            matcher = BodyMatcher(models_dir, self.cfg.state_dir / "kleding.json")
            log.info("Herkennen aan kleding/postuur aan")
            return matcher
        except Exception as exc:  # noqa: BLE001
            log.error("Model voor kleding/postuur laden mislukt (%s)", exc)
            return None

    def _load_faces(self, force: bool = False):
        if not force and not any(c.settings["faces"]["enabled"] for c in self.cfg.cameras):
            return None
        from .faces import FaceRecognizer
        models_dir = Path(self.cfg.model["path"]).parent
        if not FaceRecognizer.available(models_dir):
            log.error("Gezichtsmodellen ontbreken in %s; gezichtsherkenning uit. "
                      "Zie README: 'Gezichtsherkenning'.", models_dir)
            return None
        try:
            return FaceRecognizer(models_dir, self.cfg.state_dir / "faces")
        except Exception as exc:  # noqa: BLE001
            log.error("Gezichtsherkenning laden mislukt (%s)", exc)
            return None

    # --- instellingen uit de webpagina ---------------------------------------
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
                raise ValueError("Gezichtsmodellen ontbreken op de server")
        if values.get("plates.enabled") and self.plates is None:
            self.plates = self._load_plates(force=True)
            if self.plates is None:
                raise ValueError("Kentekenherkenning is niet geïnstalleerd op de server")
        self.cameras[cam].apply_settings(values, self.faces, self.plates)
        with self._switch_lock:
            data = self._load_json(self._settings_file)
            data.setdefault(cam, {}).update(values)
            self._save_json(self._settings_file, data)
        if self.mqtt and ("faces.enabled" in values or "plates.enabled" in values):
            self.mqtt.reannounce()      # sensor 'Gezicht' erbij of weg
        return self.camera_settings(cam)

    # --- maskers uit de webpagina --------------------------------------------
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

    # --- switches bewaren over herstarts heen ------------------------------
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
        if camera and label in ("aanbellen", "beweging"):
            camera.manual_trigger(label)

    def _state(self, cam: str) -> dict[str, bool]:
        c = self.cameras[cam]
        return {"detect": c.detect_enabled, "record": c.record_enabled, "online": c.online,
                "beweging": c.motion_active, **c.object_active, "rondhangen": c.loitering,
                "gezicht": c.face_name, "kenteken": c.plate}

    def _after_cleanup(self) -> None:
        self.store.forget_missing()
        if self.mqtt:
            for name in self.cameras:
                self.mqtt.publish_recent(name)

    # --- levensloop ----------------------------------------------------------
    def start(self) -> None:
        for cam in self.cameras.values():
            cam.start()
        if self.mqtt:
            self.mqtt.face_handler = self._on_face_answer
            self.mqtt.presence_handler = self._on_presence
            self.mqtt.extern_handler = self._on_extern_ai
            self.mqtt.start([c.cfg for c in self.cameras.values()], self._on_switch, self._state,
                            self._on_trigger)
        self.retention.start()
        self._stop_evt = threading.Event()
        if self.mqtt and self.cfg.summary_time:
            threading.Thread(target=self._summary_loop, name="dagoverzicht", daemon=True).start()
        if self.mqtt:
            threading.Thread(target=self._health_loop, name="gezondheid", daemon=True).start()
        if self.ai:
            self.ai.start()
            log.info("AI-beschrijvingen aan (%s via %s)", self.cfg.ai["model"], self.cfg.ai["base_url"])
        log.info("camera-service gestart met %d camera('s)", len(self.cameras))

    def health_snapshot(self) -> dict:
        return self.health.snapshot(self.ai.state if self.ai else "uit",
                                    sum(c.online for c in self.cameras.values()), len(self.cameras))

    def _health_loop(self) -> None:
        """Elke minuut CPU, geheugen, schijf en AI-status naar HA; na een crash één melding."""
        self.mqtt.connected.wait(120)
        if self.health.previous_crash:
            self.mqtt.publish_server_event("herstart_na_crash", {
                "gestart": self.health.started.isoformat(timespec="seconds")})
        while True:
            try:
                self.mqtt.publish_health(self.health_snapshot())
            except Exception:  # noqa: BLE001
                log.exception("Gezondheid publiceren mislukt")
            if self._stop_evt.wait(60):
                return

    def _on_extern_ai(self, cam: str, result: dict) -> None:
        """Oordeel van de externe AI (HA-script camera_ai_check) bij het event bewaren."""
        camera = self.cameras.get(cam)
        if camera is None:
            return
        result["tijd"] = datetime.now().astimezone().isoformat(timespec="seconds")
        if not camera.attach_extern(result):
            log.info("[%s] extern AI-oordeel zonder bijpassend event", cam)

    def mark_false(self, clip_rel: str, false: bool) -> dict:
        """Event als vals alarm markeren (of terugdraaien). Bij een object met een plek onthoudt
        de camera die plek, zodat hetzelfde ding daar niet opnieuw een event geeft."""
        js = self.store.json_path(clip_rel)
        if js is None:
            raise KeyError("Event niet gevonden")
        data = json.loads(js.read_text())
        camera = self.cameras.get(data.get("camera", ""))
        learned = False
        if false:
            if camera and data.get("box") and data.get("label") in camera.object_active:
                learned = camera.add_false_spot(data["label"], data["box"], data["id"])
            data = self.store.update(js, {"vals": "gebruiker"})
        else:
            if camera:
                camera.remove_false_spot(data["id"])
            data = self.store.update(js, {"vals": None})
        return {"vals": data.get("vals") if data else None, "plek_onthouden": learned}

    def _on_presence(self, name: str, home: bool) -> None:
        """HA meldt dat een bewoner thuiskomt (of weggaat)."""
        if home:
            self.arrivals[name] = time.monotonic()
            log.info("%s is thuisgekomen (volgens HA)", name)
        else:
            self.arrivals.pop(name, None)

    def _on_face_answer(self, sample: str, name: str | None, ignore: bool) -> None:
        """Antwoord op 'Wie is dit?' uit een melding: de hele groep een naam geven of negeren."""
        if self.faces is None:
            return
        group = self.faces.unknown.group_of(sample)
        if group is None:
            log.info("Gezicht %s al verwerkt of verlopen", sample)
            return
        try:
            if ignore or not name:
                self.faces.unknown.dismiss(group["ids"])
                log.info("Onbekend gezicht genegeerd vanuit melding (%d foto's)", group["count"])
                return
            updates = self.faces.unknown.assign(group["ids"], name)
            for crop, who in updates:
                self.store.set_face_name(crop, who)
            log.info("Gezicht vanuit melding geleerd als %s (%d events bijgewerkt)", name, len(updates))
        except (ValueError, KeyError) as exc:
            log.warning("Gezicht vanuit melding niet verwerkt: %s", exc)

    def _summary_loop(self) -> None:
        """Elke dag op summary_time het dagoverzicht naar HA sturen."""
        from .summary import summarize_day
        sent = None
        today_at = 0.0
        while not self._stop_evt.wait(20):
            now = datetime.now().astimezone()
            if time.monotonic() - today_at >= 300:      # tussenstand voor het HA-dashboard
                today_at = time.monotonic()
                try:
                    self.mqtt.publish_today(summarize_day(
                        self.cfg.clips_dir, now.date(), self.plates.known if self.plates else {}))
                except Exception:  # noqa: BLE001
                    log.exception("Tussenstand van vandaag maken mislukt")
            if now.strftime("%H:%M") != self.cfg.summary_time or sent == now.date():
                continue
            sent = now.date()
            try:
                s = summarize_day(self.cfg.clips_dir, now.date(), self.plates.known if self.plates else {})
                self.mqtt.publish_summary(s)
                log.info("Dagoverzicht verstuurd: %s", s["tekst"])
            except Exception:  # noqa: BLE001
                log.exception("Dagoverzicht maken mislukt")

    def stop(self) -> None:
        log.info("Stoppen…")
        if hasattr(self, "_stop_evt"):
            self._stop_evt.set()
        for cam in self.cameras.values():
            cam.stop()
        self.retention.stop()
        if self.ai:
            self.ai.stop()
        deadline = time.monotonic() + 20      # samen binnen systemd's TimeoutStopSec
        for cam in self.cameras.values():
            cam.recorder.join(timeout=max(0.5, deadline - time.monotonic()))   # clip afsluiten
        if self.mqtt:
            self.mqtt.stop()
        self.health.clean_stop()
