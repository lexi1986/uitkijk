"""Eén camera: sub-stream decoderen voor detectie, main-stream bufferen/opnemen."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import av
import cv2
import numpy as np

from .config import CameraConfig
from .detector import Detection, ObjectDetector, draw_detections
from .events import EventRecord, EventStore, clip_paths
from .faces import FaceMatch, FaceRecognizer, summarize
from .motion import MotionDetector, build_mask, motion_region, point_masked, zones_at
from .plates import PlateRead, PlateReader, vote
from .recorder import RTSP_OPTIONS, MainRecorder

if TYPE_CHECKING:
    from .ai_describe import AiDescriber
    from .mqtt import MqttBridge

log = logging.getLogger(__name__)

LABEL_PRIORITY = {"aanbellen": 4, "persoon": 3, "voertuig": 2, "dier": 1, "beweging": 0}
# Labels die stil kunnen staan zonder dat het interessant is (geparkeerde auto, slapende kat).
STATIONARY_LABELS = {"voertuig", "dier"}
STATIONARY_IOU = 0.7      # zoveel overlap met de vorige plek = niet bewogen
# Pas vergeten als een object zo lang helemaal niet meer gezien is. Niet tellen hoe vaak het
# model het mist: 's nachts schommelt de zekerheid rond de drempel en zou een geparkeerde auto
# zo steeds 'nieuw' worden. Een auto die wegrijdt of aankomt beweegt en telt dan gewoon mee.
STATIONARY_FORGET = 3600.0
FULL_FRAME_EVERY = 10.0   # minstens zo vaak het hele beeld bekijken, ook als er ingezoomd kan worden
STATIONARY_MAX_TRACKS = 30
STATIONARY_SAVE_EVERY = 30.0   # seconden tussen het bewaren van stilstaande objecten
# Een deel van een stilstaand object telt alleen mee als dat object kort geleden nog gezien is;
# een auto die wegreed en later terugkomt geeft zo gewoon weer een event.
STATIONARY_PART_RECENT = 900.0
STATIONARY_REFRESH = 300.0
# Uitsnede voor de AI-beschrijving: zoveel keer de box, minstens dit deel van de beeldhoogte.
AI_CROP_MARGIN = 2.0
AI_CROP_MIN = 0.45
# "Vals alarm" vanuit de webpagina: zelfde soort object op vrijwel dezelfde plek telt niet meer.
FALSE_SPOT_IOU = 0.6
FALSE_SPOTS_MAX = 50    # zo vaak in rust nakijken of stilstaande objecten er nog staan
# Twijfelgevallen (score onder objects.verify_below) eerst nakijken op een scherpe, ingezoomde
# uitsnede uit de main-stream. Een regenpijp of tuinzak scoort daar 0, een echt persoon 0.8+.
VERIFY_LABELS = {"persoon", "dier"}
PLATE_PARKED_AFTER = 900.0    # onbekende auto die zo lang blijft staan: melden
NIGHT_HOURS = (22, 7)         # onbekende auto tussen deze uren: meteen melden
RING_FACE_ATTEMPTS = 3
ASK_FACE_AT = (1, 3)
BODY_ATTEMPTS = 3             # zo vaak per event een scherpe uitsnede van de persoon maken
BODY_INTERVAL = 2.0
ARRIVAL_WINDOW = 300.0        # zo lang na thuiskomst (volgens HA) is een onbekende persoon waarschijnlijk die bewoner          # bij de zoveelste keer dat een onbekend gezicht langskomt: vragen wie het is    # bij aanbellen zo vaak (1 s ertussen) in het hele beeld naar een gezicht zoeken
VERIFY_MIN = 0.5          # zoveel moet de scherpe uitsnede minstens opleveren
VERIFY_KEEP_OK = 30.0     # bevestigd: zo lang niet opnieuw nakijken (zolang het in de buurt blijft)
VERIFY_KEEP_BAD = 300.0   # afgewezen plek zo lang negeren


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def contained(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Welk deel van box a binnen box b valt (0-1)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


@dataclass
class Verdict:
    """Uitkomst van het nakijken van een twijfelgeval op een plek in het beeld."""
    label: str
    box: tuple[int, int, int, int]
    until: float
    ok: bool


def zoom_score(main: np.ndarray, box: tuple[int, int, int, int], sub_shape: tuple[int, int],
               detector: ObjectDetector, label: str) -> float:
    """Hoogste score voor `label` in een vierkante uitsnede van het main-beeld rond een box
    uit de sub-stream (minstens 320 px, zodat het model niet hoeft te vergroten)."""
    sy, sx = main.shape[0] / sub_shape[0], main.shape[1] / sub_shape[1]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 * sx, (y1 + y2) / 2 * sy
    side = max(320.0, 1.6 * max((x2 - x1) * sx, (y2 - y1) * sy))
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    crop = main[b:int(b + side), a:int(a + side)]
    if crop.size == 0:
        return 0.0
    return max((d.score for d in detector.detect(crop, [label], 0.25)), default=0.0)


@dataclass
class Track:
    """Een object dat (bijna) op dezelfde plek blijft staan."""
    label: str
    box: tuple[int, int, int, int]
    first_seen: float
    last_seen: float = 0.0


@dataclass
class ActiveEvent:
    id: str
    start_wall: datetime
    start_mono: float
    last_trigger: float
    label: str
    labels: set[str] = field(default_factory=set)
    best_score: float | None = None
    snapshot_jpg: bytes | None = None
    faces: list[FaceMatch] = field(default_factory=list)
    face_attempts: int = 0
    last_face_try: float = 0.0
    plates: list[PlateRead] = field(default_factory=list)
    ai_requested: bool = False
    zones: set[str] = field(default_factory=set)
    ai: dict | None = None
    plate_attempts: int = 0
    vehicle_box: tuple[int, int, int, int] | None = None   # laatste plek van het voertuig
    body_emb: np.ndarray | None = None    # kleding/postuur van de grootste persoon (her-identificatie)
    body_area: int = 0
    body_attempts: int = 0
    last_body_try: float = 0.0
    max_persons: int = 0
    last_plate_try: float = 0.0
    best_box: list[float] | None = None   # beste detectie als fractie van het beeld
    ai_labels: set[str] = field(default_factory=set)   # labels toen de AI-beschrijving gevraagd werd
    vals: str | None = None
    extern: dict | None = None


class Camera:
    def __init__(self, cfg: CameraConfig, clips_dir: Path, state_dir: Path,
                 detector: ObjectDetector | None, store: EventStore,
                 mqtt: "MqttBridge | None", switches: dict[str, bool],
                 on_switch_change: Callable[[], None], faces: FaceRecognizer | None = None,
                 plates: PlateReader | None = None, ai: "AiDescriber | None" = None):
        self.cfg = cfg
        self.name = cfg.name
        self.clips_dir = clips_dir
        self.detector = detector
        self.store = store
        self.mqtt = mqtt
        self.detect_enabled = switches.get("detect", True)
        self.record_enabled = switches.get("record", True)
        self._on_switch_change = on_switch_change

        self.main_url = cfg.main_url
        self.sub_url = cfg.sub_url or cfg.main_url
        self._resolve_lock = threading.Lock()

        self.motion = MotionDetector(cfg.motion["threshold"], cfg.motion["min_area"], cfg.mask)
        self.recorder = MainRecorder(
            cfg.name, self._get_main_url, cfg.pre_seconds, cfg.settings["record_audio"],
            clips_dir / ".parts" / cfg.name, self._on_clip_done, self._on_main_online)

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._sub_thread = threading.Thread(target=self._sub_loop, name=f"sub-{cfg.name}", daemon=True)
        self._tick_thread = threading.Thread(target=self._tick_loop, name=f"tick-{cfg.name}", daemon=True)

        # toestand
        self.sub_online = False
        self.motion_active = False
        self._last_motion = 0.0
        self.object_active: dict[str, bool] = {lbl: False for lbl in ("persoon", "voertuig", "dier")}
        self._last_object: dict[str, float] = {lbl: 0.0 for lbl in self.object_active}
        self._hits: dict[str, deque[float]] = {lbl: deque() for lbl in self.object_active}
        self._tracks: list[Track] = []
        self._tracks_file = state_dir / f"stationary_{cfg.name}.json"
        self._tracks_saved = 0.0
        self._person_since: float | None = None
        self.loitering = False
        self._last_infer = 0.0
        self._last_full_infer = 0.0
        self._last_refresh = 0.0
        self.event: ActiveEvent | None = None
        self._pending: dict[str, ActiveEvent] = {}  # gestopt, wacht op de clip van de recorder
        self.latest_bgr: np.ndarray | None = None
        self.latest_snapshot_jpg: bytes | None = None
        self.stats = {"decode_fps": 0.0, "motion_fps": 0.0, "infer_ms": 0.0,
                      "hwaccel": "none"}

        self.faces = faces if cfg.settings["faces"]["enabled"] else None
        self._face_busy = False
        self.face_name = "geen"     # laatst herkende persoon (naam / onbekend / geen)
        self.plates = plates if cfg.settings["plates"]["enabled"] else None
        self._plate_busy = False
        self.plate = "geen"         # laatst gelezen kenteken (of geen)
        self._verdicts: list[Verdict] = []
        self._zone_hits: set[str] = set()
        self._plate_watch: list[dict] = []
        self.arrivals: dict[str, float] = {}  # naam -> monotonic tijd van thuiskomst (gedeeld, van de service)
        self.bodies = None                    # BodyMatcher (kleding/postuur), gezet door de service
        self._body_busy = False    # onbekende auto's: staan ze er na een kwartier nog?   # zones met een object sinds het vorige event-moment
        self._verify_busy = False
        self.ai = ai
        self._saved_json: dict[str, Path] = {}   # event-id -> JSON, voor late AI-resultaten
        self._false_file = state_dir / f"vals_{cfg.name}.json"
        self.false_spots: list[dict] = self._load_false_spots()

        self._apply_rates()

    def _apply_rates(self) -> None:
        """Hoe vaak het model mag draaien; opnieuw na een wijziging van de instellingen."""
        accelerated = self.detector is not None and self.detector.accelerated
        max_fps = self.cfg.objects["max_fps"]
        self.max_infer_fps = (5.0 if accelerated else 2.0) if str(max_fps) == "auto" else float(max_fps)
        self.always_detect = bool(self.cfg.objects["always"])
        # zonder beweging rustig kijken: op de CPU 1x per seconde, alleen naar personen
        self.idle_infer_fps = self.max_infer_fps if accelerated else 1.0

    # ------------------------------------------------------------------ start/stop
    def start(self) -> None:
        self.store.load_recent(self.name)
        self._load_tracks()
        self.recorder.start()
        self._sub_thread.start()
        self._tick_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.recorder.stop()
        self.recorder.stop_event()

    # ------------------------------------------------------------------ URL's (ONVIF)
    def _resolve_urls(self) -> None:
        with self._resolve_lock:
            if self.main_url:
                return
            from .onvif_discovery import discover_streams
            o = self.cfg.onvif
            main, sub = discover_streams(o["host"], o["username"], o["password"], int(o.get("port", 80)))
            self.main_url = main
            self.sub_url = self.cfg.sub_url or sub
            log.info("[%s] streams via ONVIF gevonden", self.name)

    def _get_main_url(self) -> str:
        self._resolve_urls()
        return self.main_url  # type: ignore[return-value]

    def _get_sub_url(self) -> str:
        self._resolve_urls()
        return self.sub_url  # type: ignore[return-value]

    # ------------------------------------------------------------------ switches
    def manual_trigger(self, label: str = "aanbellen") -> bool:
        """Event starten of verlengen van buitenaf (bijv. de deurbelknop via HA)."""
        if not self.recorder.online:
            return False
        with self._lock:
            self._trigger(time.monotonic(), label, None, None, self.latest_bgr, [])
            ev = self.event
        log.info("[%s] handmatige trigger: %s", self.name, label)
        if label == "aanbellen" and self.faces is not None and ev is not None:
            # niet wachten op de persoonsdetectie: meteen in het hele scherpe beeld zoeken,
            # zodat de melding binnen een paar seconden een naam kan krijgen
            threading.Thread(target=self._ring_faces, args=(ev.id,),
                             name=f"ring-{self.name}", daemon=True).start()
        return True

    def _ring_faces(self, event_id: str) -> None:
        for attempt in range(RING_FACE_ATTEMPTS):
            if attempt:
                time.sleep(1.0)
            with self._lock:
                ev = self.event if self.event and self.event.id == event_id else None
                busy = self._face_busy
                if ev is None or any(f.name and f.score >= self.faces.backend.learn for f in ev.faces):
                    return
                if not busy:
                    self._face_busy = True
                    ev.face_attempts += 1
            if not busy:
                self._face_job(event_id, (1, 1), [])

    def apply_settings(self, values: dict, faces: FaceRecognizer | None,
                       plates: PlateReader | None = None) -> None:
        """Instellingen uit de webpagina (al gecontroleerd), direct actief."""
        from .config import set_path
        with self._lock:
            for path, value in values.items():
                set_path(self.cfg.settings, path, value)
            self.motion = MotionDetector(self.cfg.motion["threshold"], self.cfg.motion["min_area"],
                                         self.cfg.mask)
            self.recorder.ring.pre_seconds = self.cfg.pre_seconds
            self.faces = faces if self.cfg.settings["faces"]["enabled"] else None
            self.plates = plates if self.cfg.settings["plates"]["enabled"] else None
            self._apply_rates()
        log.info("[%s] instellingen bijgewerkt: %s", self.name, ", ".join(values))

    def set_mask(self, polygons: list[list[list[float]]]) -> None:
        """Nieuw masker (al gecontroleerd), direct actief voor beweging en detectie."""
        with self._lock:
            self.cfg.settings["mask"] = polygons
            self.motion = MotionDetector(self.cfg.motion["threshold"], self.cfg.motion["min_area"],
                                         polygons)
        log.info("[%s] masker ingesteld: %d vlak(ken)", self.name, len(polygons))

    def set_zones(self, zones: list[dict]) -> None:
        """Nieuwe zones (al gecontroleerd), direct actief voor de objectdetectie."""
        with self._lock:
            self.cfg.settings["zones"] = zones
        log.info("[%s] zones ingesteld: %s", self.name, ", ".join(z["name"] for z in zones) or "geen")

    def set_switch(self, which: str, on: bool) -> None:
        with self._lock:
            if which == "detect":
                self.detect_enabled = on
                if not on:
                    self._clear_sensors()
            elif which == "record":
                self.record_enabled = on
                if not on and self.event is not None:
                    self._end_event()
        log.info("[%s] %s %s", self.name, which, "aan" if on else "uit")
        if self.mqtt:
            self.mqtt.publish_switch(self.name, which, on)
        self._on_switch_change()

    def _clear_sensors(self) -> None:
        if self.motion_active:
            self.motion_active = False
            self._publish_binary("beweging", False)
        for lbl, on in self.object_active.items():
            if on:
                self.object_active[lbl] = False
                self._publish_binary(lbl, False)
            self._hits[lbl].clear()

    # ------------------------------------------------------------------ sub-stream
    def _hw_device_type(self) -> str | None:
        """Welke hardware-decoder te gebruiken, of None voor CPU."""
        wanted = str(self.cfg.settings["hwaccel"]).lower()
        if wanted in ("none", "cpu", "off", ""):
            return None
        if not Path("/dev/dri/renderD128").exists():
            return None
        from av.codec.hwaccel import hwdevices_available
        available = hwdevices_available()
        for dev in (["vaapi", "qsv"] if wanted == "auto" else [wanted]):
            if dev in available:
                return dev
        if not getattr(self, "_hw_warned", False):
            self._hw_warned = True
            log.warning("[%s] hardware-decoding '%s' niet beschikbaar in deze PyAV-build "
                        "(wel: %s); de sub-stream wordt op de CPU gedecodeerd",
                        self.name, wanted, ", ".join(available) or "geen")
        return None

    def _open_sub(self):
        url = self._get_sub_url()
        dev = self._hw_device_type()
        if dev:
            try:
                from av.codec.hwaccel import HWAccel
                hw = HWAccel(dev, device="/dev/dri/renderD128", allow_software_fallback=True)
                c = av.open(url, options=RTSP_OPTIONS, timeout=(10.0, 10.0), hwaccel=hw)
                self.stats["hwaccel"] = dev
                return c
            except Exception as exc:  # noqa: BLE001
                log.warning("[%s] %s niet bruikbaar, terug naar CPU: %s", self.name, dev, exc)
        self.stats["hwaccel"] = "none"
        return av.open(url, options=RTSP_OPTIONS, timeout=(10.0, 10.0))

    def _sub_loop(self) -> None:
        backoff = 2.0
        while not self._stop.is_set():
            try:
                with self._open_sub() as container:
                    stream = container.streams.video[0]
                    stream.thread_type = "AUTO"
                    self._set_sub_online(True)
                    backoff = 2.0
                    self._process_frames(container, stream)
            except Exception as exc:  # noqa: BLE001
                log.warning("[%s] sub-stream fout: %s", self.name, exc)
            self._set_sub_online(False)
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 2, 60.0)

    def _set_sub_online(self, ok: bool) -> None:
        if ok != self.sub_online:
            self.sub_online = ok
            log.info("[%s] sub-stream %s", self.name, "verbonden" if ok else "offline")
            if not ok:
                with self._lock:
                    self._clear_sensors()
            self._publish_online()

    def _process_frames(self, container, stream) -> None:
        interval = 1.0 / float(self.cfg.motion["fps"])
        width = int(self.cfg.motion["width"])
        next_at = 0.0
        decoded = processed = 0
        window_start = time.monotonic()
        for frame in container.decode(stream):
            if self._stop.is_set():
                return
            decoded += 1
            now = time.monotonic()
            if now - window_start >= 5:
                self.stats["decode_fps"] = round(decoded / (now - window_start), 1)
                self.stats["motion_fps"] = round(processed / (now - window_start), 1)
                decoded = processed = 0
                window_start = now
            if now < next_at:
                continue
            next_at = now + interval
            processed += 1
            if not self.detect_enabled:
                continue
            height = int(round(frame.height * width / frame.width / 2) * 2)
            gray = frame.reformat(width=width, height=height, format="gray").to_ndarray()
            fraction = self.motion.detect(gray)
            moving = self.motion.is_motion(fraction)
            bgr = None
            recent_motion = now - self._last_motion <= float(self.cfg.motion["cooldown_seconds"])
            idle_due = self.always_detect and now - self._last_infer >= 1.0 / self.idle_infer_fps
            need_bgr = (moving or recent_motion or idle_due or self.latest_bgr is None
                        or processed % 25 == 0)
            if need_bgr:
                bgr = frame.to_ndarray(format="bgr24")
                self.latest_bgr = bgr
            self._on_frame(now, moving, bgr)

    # ------------------------------------------------------------------ detectielogica
    def _on_frame(self, now: float, moving: bool, bgr: np.ndarray | None) -> None:
        cfg = self.cfg
        if bgr is not None and not moving and self._refresh_due(now):
            self._refresh_parked(now, bgr)
        with self._lock:
            if moving:
                self._last_motion = now
                if not self.motion_active:
                    self.motion_active = True
                    self._publish_binary("beweging", True)

            labels = cfg.enabled_labels()
            use_objects = cfg.objects["enabled"] and self.detector is not None and bool(labels)
            if not use_objects:
                if moving:
                    self._trigger(now, "beweging", None, None, bgr, [])
                return

            if bgr is None:
                return
            in_motion = now - self._last_motion <= cfg.motion["cooldown_seconds"]
            if not in_motion:
                # Zonder beweging telt alleen een persoon (iemand die stil voor de deur staat);
                # een geparkeerde auto of slapende kat zou anders eindeloos events maken.
                if not (self.always_detect and "persoon" in labels):
                    return
                labels = ["persoon"]
            fps = self.max_infer_fps if in_motion else self.idle_infer_fps
            if now - self._last_infer < 1.0 / fps:
                return
            self._last_infer = now
            # inzoomen op de beweging: kleine of verre objecten worden zo veel beter gevonden
            region = None
            if in_motion and self.motion.last_box and now - self._last_full_infer < FULL_FRAME_EVERY:
                region = motion_region(self.motion.last_box, bgr.shape[:2])
            if region is None:
                self._last_full_infer = now

        t0 = time.perf_counter()
        if region is None:
            detections = self.detector.detect(bgr, labels, float(cfg.objects["min_score"]))
        else:
            x0, y0, x1, y1 = region
            detections = [Detection(d.label, d.score, (d.box[0] + x0, d.box[1] + y0, d.box[2] + x0, d.box[3] + y0))
                          for d in self.detector.detect(bgr[y0:y1, x0:x1], labels,
                                                        float(cfg.objects["min_score"]))]
            self.stats["zoom"] = self.stats.get("zoom", 0) + 1
        infer_ms = (time.perf_counter() - t0) * 1000
        self.stats["infer_ms"] = round(0.8 * self.stats["infer_ms"] + 0.2 * infer_ms, 1)

        if cfg.mask:
            mask = build_mask(cfg.mask, bgr.shape[1], bgr.shape[0])
            detections = [d for d in detections
                          if not point_masked(mask, (d.box[0] + d.box[2]) / 2, d.box[3])]

        if cfg.zones:
            # alleen objecten die met hun onderkant in een zone staan tellen mee
            zoned = []
            for d in detections:
                names = zones_at(cfg.zones, bgr.shape[1], bgr.shape[0],
                                 (d.box[0] + d.box[2]) / 2, d.box[3])
                if names:
                    zoned.append(d)
                    self._zone_hits.update(names)
            detections = zoned
        if self.false_spots:
            detections = [d for d in detections if not self._at_false_spot(d, bgr.shape[:2])]
        detections = self._verify(now, detections, bgr.shape[:2])
        with self._lock:
            detections = self._drop_stationary(now, detections, labels)
            window = float(cfg.objects["hit_window_seconds"])
            for d in detections:
                self._hits[d.label].append(now)
            for lbl, hits in self._hits.items():
                while hits and hits[0] < now - window:
                    hits.popleft()
            confirmed = [lbl for lbl, hits in self._hits.items()
                         if len(hits) >= int(cfg.objects["min_hits"])]
            for lbl in confirmed:
                self._last_object[lbl] = now
                if not self.object_active[lbl]:
                    self.object_active[lbl] = True
                    self._publish_binary(lbl, True)
                    if lbl == "persoon":
                        self._person_since = now
            if confirmed:
                best = max((d for d in detections if d.label in confirmed),
                           key=lambda d: d.score, default=None)
                label = max(confirmed, key=lambda lbl: LABEL_PRIORITY[lbl])
                self._trigger(now, label, best.score if best else None, best, bgr, detections)
                if "persoon" in confirmed:
                    people = [d.box for d in detections if d.label == "persoon"]
                    if self.event is not None:
                        self.event.max_persons = max(self.event.max_persons, len(people))
                    self._maybe_faces(now, bgr.shape[:2], people)
                    self._maybe_body(now, bgr.shape[:2], people)
                if "voertuig" in confirmed:
                    cars = [d.box for d in detections if d.label == "voertuig"]
                    if cars and self.event is not None:
                        self.event.vehicle_box = max(cars, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                    self._maybe_plates(now, bgr.shape[:2], cars)

    def _drop_stationary(self, now: float, detections: list[Detection],
                         labels: list[str]) -> list[Detection]:
        """Laat voertuigen en dieren weg die al `stationary_seconds` op dezelfde plek staan
        (lock is al genomen). Zo geeft een geparkeerde auto geen nieuwe events zodra er
        ergens anders in beeld iets beweegt."""
        limit = float(self.cfg.objects["stationary_seconds"])
        if limit <= 0:
            return detections
        matched: set[int] = set()
        moving = []
        for d in detections:
            if d.label not in STATIONARY_LABELS:
                moving.append(d)
                continue
            best = max((t for t in self._tracks if t.label == d.label and id(t) not in matched),
                       key=lambda t: iou(t.box, d.box), default=None)
            if best is not None and iou(best.box, d.box) >= STATIONARY_IOU:
                matched.add(id(best))
                best.box, best.last_seen = d.box, now
                if now - best.first_seen >= limit:
                    continue            # staat stil: geen trigger
            elif self._part_of_parked(now, d, limit):
                continue                # zonnige zijkant of schaduwrand van een geparkeerde auto
            else:
                track = Track(d.label, d.box, now, now)
                self._tracks.append(track)
                matched.add(id(track))
            moving.append(d)
        self._tracks = [t for t in self._tracks
                        if now - t.last_seen < STATIONARY_FORGET][-STATIONARY_MAX_TRACKS:]
        return moving

    def _refresh_due(self, now: float) -> bool:
        return (self.detector is not None and self.cfg.objects["enabled"] and bool(self._tracks)
                and now - self._last_motion > float(self.cfg.motion["cooldown_seconds"])
                and now - self._last_refresh >= STATIONARY_REFRESH)

    def _refresh_parked(self, now: float, bgr: np.ndarray) -> None:
        """Zonder beweging draait de detectie niet, waardoor een geparkeerde auto 'niet gezien'
        raakt. Dan herkent _part_of_parked een deelbox ervan (een langsrijdende auto door de
        ruiten, zon op de zijkant) niet meer. Daarom af en toe in rust natellen wat er staat."""
        self._last_refresh = now
        detections = self.detector.detect(bgr, sorted(STATIONARY_LABELS),
                                          float(self.cfg.objects["min_score"]))
        with self._lock:
            self._mark_seen(now, detections)

    def _mark_seen(self, now: float, detections: list[Detection]) -> None:
        """Stilstaande objecten die nog op hun plek staan als gezien markeren (lock is al genomen)."""
        for d in detections:
            for t in self._tracks:
                if t.label == d.label and iou(t.box, d.box) >= STATIONARY_IOU:
                    t.last_seen = now

    def _part_of_parked(self, now: float, d: Detection, limit: float) -> bool:
        """Ziet het model maar een deel van een stilstaand object (felle zon op de zijkant,
        schaduw over de motorkap), dan is de box veel kleiner en telt hij met IoU als nieuw.
        Valt de box grotendeels binnen een kort geleden nog geziene stilstaande auto, of
        omvat hij die bijna precies (auto + schaduw), dan is het dezelfde."""
        for t in self._tracks:
            if (t.label != d.label or now - t.first_seen < limit
                    or now - t.last_seen > STATIONARY_PART_RECENT):
                continue
            if contained(d.box, t.box) >= 0.8 or (contained(t.box, d.box) >= 0.8
                                                 and iou(t.box, d.box) >= 0.5):
                t.last_seen = now
                return True
        return False

    # ------------------------------------------------------------------ twijfelgevallen
    def _verify(self, now: float, detections: list[Detection],
                sub_shape: tuple[int, int]) -> list[Detection]:
        """Laat twijfelgevallen pas meetellen als de scherpe uitsnede ze bevestigt. Tot die
        uitkomst er is (± een halve seconde) tellen ze niet mee."""
        below = float(self.cfg.objects.get("verify_below", 0) or 0)
        if below <= 0:
            return detections
        with self._lock:
            self._verdicts = [v for v in self._verdicts if v.until > now]
            verdicts = list(self._verdicts)
        kept, todo = [], []
        for d in detections:
            if d.label not in VERIFY_LABELS or d.score >= below:
                kept.append(d)
                continue
            same = [v for v in verdicts if v.label == d.label]
            if any(v.ok and iou(v.box, d.box) > 0.1 for v in same):
                kept.append(d)
            elif not any(not v.ok and iou(v.box, d.box) >= 0.4 for v in same):
                todo.append(d)
        if todo and not self._verify_busy:
            self._verify_busy = True
            threading.Thread(target=self._verify_job, args=(todo, sub_shape),
                             name=f"verify-{self.name}", daemon=True).start()
        return kept

    def _verify_job(self, todo: list[Detection], sub_shape: tuple[int, int]) -> None:
        try:
            frame = self.recorder.grab_frame()
            now = time.monotonic()
            found = []
            for d in todo:
                # geen scherp beeld beschikbaar: niet strenger zijn dan zonder controle
                score = 1.0 if frame is None else zoom_score(frame, d.box, sub_shape,
                                                             self.detector, d.label)
                ok = score >= VERIFY_MIN
                found.append(Verdict(d.label, d.box, now + (VERIFY_KEEP_OK if ok else VERIFY_KEEP_BAD), ok))
                if not ok:
                    self.stats["verify_rejected"] = self.stats.get("verify_rejected", 0) + 1
                    log.info("[%s] twijfelgeval %s (%.2f) op %s afgewezen: scherp beeld %.2f",
                             self.name, d.label, d.score, d.box, score)
            with self._lock:
                self._verdicts = (self._verdicts + found)[-50:]
        except Exception:  # noqa: BLE001 - nakijken mag de camera nooit hinderen
            log.exception("[%s] nakijken twijfelgeval mislukt", self.name)
        finally:
            self._verify_busy = False

    # ------------------------------------------------------------------ gezichten
    def _maybe_faces(self, now: float, sub_shape: tuple[int, int],
                     boxes: list[tuple[int, int, int, int]]) -> None:
        """Start op de achtergrond een gezichtsherkenning (lock is al genomen)."""
        ev = self.event
        fcfg = self.cfg.settings["faces"]
        if (self.faces is None or ev is None or not boxes or self._face_busy
                or ev.face_attempts >= int(fcfg["attempts"])
                or now - ev.last_face_try < float(fcfg["interval_seconds"])
                or any(f.name and f.score >= self.faces.backend.learn for f in ev.faces)):
            return                  # pas stoppen als iemand zeker herkend is
        ev.face_attempts += 1
        ev.last_face_try = now
        self._face_busy = True
        threading.Thread(target=self._face_job, args=(ev.id, sub_shape, boxes),
                         name=f"faces-{self.name}", daemon=True).start()

    def _face_job(self, event_id: str, sub_shape: tuple[int, int],
                  boxes: list[tuple[int, int, int, int]]) -> None:
        try:
            frame = None
            if self.cfg.settings["faces"]["source"] == "main":
                frame = self.recorder.grab_frame()
            if frame is None and self.latest_bgr is not None:
                frame = self.latest_bgr.copy()
            if frame is None:
                return
            sy, sx = frame.shape[0] / sub_shape[0], frame.shape[1] / sub_shape[1]
            matches: list[FaceMatch] = []
            if not boxes:                       # geen persoon bekend (aanbellen): hele beeld
                matches = self.faces.analyse(frame)
            for x1, y1, x2, y2 in boxes:
                # bovenste deel van de persoon, iets ruimer: daar zit het hoofd
                bw, bh = x2 - x1, y2 - y1
                cx1 = int(max(0, (x1 - 0.2 * bw) * sx))
                cx2 = int(min(frame.shape[1], (x2 + 0.2 * bw) * sx))
                cy1 = int(max(0, (y1 - 0.1 * bh) * sy))
                cy2 = int(min(frame.shape[0], (y1 + 0.6 * bh) * sy))
                if cx2 - cx1 < 40 or cy2 - cy1 < 40:
                    continue
                matches += self.faces.analyse(frame[cy1:cy2, cx1:cx2])
            if not matches:
                return
            with self._lock:
                ev = self.event if self.event and self.event.id == event_id else None
                if ev is not None:
                    # alle pogingen samen bewaren (beste beelden eerst), de keuze volgt per persoon
                    ev.faces = sorted(ev.faces + matches, key=lambda m: -m.quality)[:16]
                    matches = ev.faces
            groups = summarize(matches, self.faces.backend.group)
            known = next((g for g in groups if g.name), None)
            name = known.name if known else "onbekend"
            log.info("[%s] gezicht: %s%s", self.name, name,
                     f" ({known.score:.2f}, {len(known.members)} beeld(en))" if known else "")
            self._set_face(name)
        except Exception:  # noqa: BLE001 - herkenning mag de camera nooit hinderen
            log.exception("[%s] gezichtsherkenning mislukt", self.name)
        finally:
            self._face_busy = False

    # ------------------------------------------------------------------ kleding en postuur
    def _maybe_body(self, now: float, sub_shape: tuple[int, int],
                    boxes: list[tuple[int, int, int, int]]) -> None:
        """(lock is al genomen) Kenmerk van de grootste persoon uit het scherpe beeld halen."""
        ev = self.event
        if (self.bodies is None or ev is None or not boxes or self._body_busy
                or ev.body_attempts >= BODY_ATTEMPTS or now - ev.last_body_try < BODY_INTERVAL):
            return
        ev.body_attempts += 1
        ev.last_body_try = now
        self._body_busy = True
        box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        threading.Thread(target=self._body_job, args=(ev.id, sub_shape, box),
                         name=f"body-{self.name}", daemon=True).start()

    def _body_job(self, event_id: str, sub_shape: tuple[int, int], box: tuple[int, int, int, int]) -> None:
        try:
            frame = self.recorder.grab_frame()
            if frame is None:
                return
            sy, sx = frame.shape[0] / sub_shape[0], frame.shape[1] / sub_shape[1]
            x1, y1, x2, y2 = int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy)
            emb = self.bodies.embed(frame[y1:y2, x1:x2])
            if emb is None:
                return
            area = (x2 - x1) * (y2 - y1)
            with self._lock:
                ev = self.event if self.event and self.event.id == event_id else None
                if ev is not None and area > ev.body_area:       # grootste (dichtstbijzijnde) beeld telt
                    ev.body_emb, ev.body_area = emb, area
        except Exception:  # noqa: BLE001 - mag de camera nooit hinderen
            log.exception("[%s] kleding/postuur bepalen mislukt", self.name)
        finally:
            self._body_busy = False

    # ------------------------------------------------------------------ kentekens
    def _maybe_plates(self, now: float, sub_shape: tuple[int, int],
                      boxes: list[tuple[int, int, int, int]]) -> None:
        """Start op de achtergrond een kentekenlezing (lock is al genomen)."""
        ev = self.event
        pcfg = self.cfg.settings["plates"]
        if (self.plates is None or ev is None or not boxes or self._plate_busy
                or ev.plate_attempts >= int(pcfg["attempts"])
                or now - ev.last_plate_try < float(pcfg["interval_seconds"])):
            return
        # twee keer hetzelfde kenteken gelezen: zeker genoeg
        best = vote(ev.plates)
        if best and sum(r.plate == best.plate for r in ev.plates) >= 2:
            return
        ev.plate_attempts += 1
        ev.last_plate_try = now
        self._plate_busy = True
        threading.Thread(target=self._plate_job, args=(ev.id, sub_shape, boxes),
                         name=f"plates-{self.name}", daemon=True).start()

    def _plate_job(self, event_id: str, sub_shape: tuple[int, int],
                   boxes: list[tuple[int, int, int, int]]) -> None:
        try:
            frame = self.recorder.grab_frame()
            if frame is None:
                return
            sy, sx = frame.shape[0] / sub_shape[0], frame.shape[1] / sub_shape[1]
            reads: list[PlateRead] = []
            for x1, y1, x2, y2 in boxes:
                bw, bh = x2 - x1, y2 - y1
                cx1 = int(max(0, (x1 - 0.1 * bw) * sx))
                cx2 = int(min(frame.shape[1], (x2 + 0.1 * bw) * sx))
                cy1 = int(max(0, (y1 - 0.1 * bh) * sy))
                cy2 = int(min(frame.shape[0], (y2 + 0.1 * bh) * sy))
                if cx2 - cx1 >= 60 and cy2 - cy1 >= 40:
                    reads += self.plates.read(frame[cy1:cy2, cx1:cx2])
            if not reads:
                return
            with self._lock:
                ev = self.event if self.event and self.event.id == event_id else None
                if ev is None:
                    return
                ev.plates = (ev.plates + reads)[-12:]
                best = vote(ev.plates)
            name = self.plates.name_of(best.plate)
            log.info("[%s] kenteken: %s%s (%.2f)", self.name, best.plate,
                     f" = {name}" if name else "", best.confidence)
            self._set_plate(best.plate, name)
        except Exception:  # noqa: BLE001 - herkenning mag de camera nooit hinderen
            log.exception("[%s] kentekenherkenning mislukt", self.name)
        finally:
            self._plate_busy = False

    def _set_plate(self, plate: str, name: str | None) -> None:
        self.plate = plate
        if self.mqtt and self.plates is not None:
            self.mqtt.publish_plate(self.name, plate, name)

    def _set_face(self, name: str) -> None:
        if name != self.face_name or name != "geen":
            self.face_name = name
            if self.mqtt and self.faces is not None:
                self.mqtt.publish_face(self.name, name)

    # ------------------------------------------------------------------ stilstand bewaren
    def _save_tracks(self, now: float) -> None:
        """Stilstaande objecten bewaren, zodat een herstart geen nieuwe events geeft."""
        self._tracks_saved = now
        data = [{"label": t.label, "box": list(t.box), "age": round(now - t.first_seen, 1),
                 "seen_ago": round(now - t.last_seen, 1)}
                for t in self._tracks]
        try:
            if data or self._tracks_file.exists():
                tmp = self._tracks_file.with_suffix(".tmp")
                tmp.write_text(json.dumps(data))
                tmp.replace(self._tracks_file)
        except OSError as exc:
            log.warning("[%s] stilstaande objecten bewaren mislukt: %s", self.name, exc)

    def _load_tracks(self) -> None:
        try:
            data = json.loads(self._tracks_file.read_text())
        except (FileNotFoundError, ValueError):
            return
        now = time.monotonic()
        with self._lock:
            self._tracks = [Track(d["label"], tuple(d["box"]), now - float(d["age"]),
                                  now - float(d.get("seen_ago", 0)))
                            for d in data if d.get("label") in STATIONARY_LABELS]
        if self._tracks:
            log.info("[%s] %d stilstaande object(en) van vorige keer onthouden",
                     self.name, len(self._tracks))

    def _trigger(self, now: float, label: str, score: float | None, best: Detection | None,
                 bgr: np.ndarray | None, detections: list[Detection]) -> None:
        """Start of verleng een event (lock is al genomen)."""
        snapshot = None
        better = score is not None and (self.event is None or self.event.best_score is None
                                        or score > self.event.best_score)
        if bgr is not None and (self.event is None or better):
            img = draw_detections(bgr, [d for d in detections if d.label == label]) if detections else bgr
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                snapshot = buf.tobytes()
                self.latest_snapshot_jpg = snapshot
                if self.mqtt:
                    self.mqtt.publish_snapshot(self.name, snapshot)

        if not self.record_enabled:
            return
        if self.event is None:
            self.event = ActiveEvent(
                id=uuid.uuid4().hex[:12], start_wall=datetime.now().astimezone(), start_mono=now,
                last_trigger=now, label=label, labels={label}, best_score=score, snapshot_jpg=snapshot)
            self.recorder.start_event(self.event.id)
            self._write_part_meta(self.event)
            log.info("[%s] event gestart: %s", self.name, label)
        else:
            ev = self.event
            ev.last_trigger = now
            ev.labels.add(label)
            if LABEL_PRIORITY[label] > LABEL_PRIORITY[ev.label]:
                ev.label = label
            if snapshot is not None:
                ev.snapshot_jpg = snapshot
                ev.best_score = score
        if snapshot is not None and best is not None and best.label == label and bgr is not None:
            h, w = bgr.shape[:2]
            self.event.best_box = [round(best.box[0] / w, 4), round(best.box[1] / h, 4),
                                   round(best.box[2] / w, 4), round(best.box[3] / h, 4)]
        self.event.zones |= self._zone_hits
        self._zone_hits = set()
        self._maybe_ai(label, best.box if best and best.label == label else None,
                       bgr.shape[:2] if bgr is not None else None)

    def _write_part_meta(self, ev: ActiveEvent) -> None:
        """Naast de halve clip noteren wat het event is, zodat de clip na een crash te redden is."""
        try:
            d = self.recorder.part_dir
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{ev.id}.meta.json").write_text(json.dumps({
                "id": ev.id, "camera": self.name, "label": ev.label,
                "start": ev.start_wall.isoformat(timespec="seconds")}))
            if ev.snapshot_jpg:
                (d / f"{ev.id}.jpg").write_bytes(ev.snapshot_jpg)
        except OSError as exc:
            log.warning("[%s] event-gegevens voor herstel niet bewaard: %s", self.name, exc)

    # ------------------------------------------------------------------ AI-beschrijving
    def _maybe_ai(self, label: str, box: tuple[int, int, int, int] | None = None,
                  box_shape: tuple[int, int] | None = None) -> None:
        """Eén beschrijving per event aanvragen (lock is al genomen); nooit blokkerend."""
        ev = self.event
        if self.ai is None or ev is None or ev.ai_requested or not self.ai.wants(self.name, label):
            return
        ev.ai_requested = True
        ev.ai_labels = set(ev.labels)
        threading.Thread(target=self._ai_job, args=(ev.id, label, ev.start_wall, box, box_shape),
                         name=f"ai-{self.name}", daemon=True).start()

    def _ai_job(self, event_id: str, label: str, start_wall: datetime,
                box: tuple[int, int, int, int] | None, box_shape: tuple[int, int] | None) -> None:
        from .ai_describe import AiJob, crop_around
        try:
            frame = self.recorder.grab_frame()       # scherp beeld zonder detectiekaders
            if frame is None and self.latest_bgr is not None:
                frame = self.latest_bgr.copy()
            if frame is None:
                return
            if self.cfg.mask:                        # masker op het hele beeld, vóór het uitsnijden
                frame[build_mask(self.cfg.mask, frame.shape[1], frame.shape[0]) == 0] = 0
            if box is not None and box_shape is not None:
                # ruim uitsnijden: met context (straat, deur, auto) herkent het model meer en
                # beoordeelt het beeld niet als "onbruikbaar" zoals bij een krappe uitsnede
                frame = crop_around(frame, box, box_shape, margin=AI_CROP_MARGIN,
                                    min_side=int(AI_CROP_MIN * frame.shape[0]))
            edge = int(self.ai.cfg["image_max_edge"])
            scale = min(1.0, edge / max(frame.shape[:2]))
            if scale < 1:                            # meteen klein: geen 4K-buffers in de wachtrij
                frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                                   interpolation=cv2.INTER_AREA)
            self.ai.submit(AiJob(event_id, self.name, label, start_wall, time.monotonic(), frame,
                                 []))           # masker is hierboven al toegepast
        except Exception:  # noqa: BLE001 - AI mag de camera nooit hinderen
            log.exception("[%s] AI-beeld klaarzetten mislukt", self.name)

    def attach_ai(self, event_id: str, result: dict) -> None:
        """Resultaat van de AI-worker bij het juiste event bewaren (ook als de clip al klaar is).
        Ziet de AI niets van wat de detectie zag, dan is het event waarschijnlijk vals."""
        from .ai_describe import false_alarm
        vals = None
        with self._lock:
            ev = self.event if self.event and self.event.id == event_id else self._pending.get(event_id)
            if ev is not None:
                ev.ai = result
                if ev.vals is None and false_alarm(ev.ai_labels or ev.labels, result):
                    ev.vals = vals = "ai"
                    if ev is self.event and self.loitering:
                        self.loitering = False
                        self._publish_binary("rondhangen", False)
            js = self._saved_json.get(event_id) if ev is None else None
        if js is not None and js.exists():
            try:
                labels = json.loads(js.read_text()).get("labels") or []
                fields = {"ai": result}
                if false_alarm(labels, result):
                    fields["vals"] = vals = "ai"
                if self.store.update(js, fields) is None:
                    raise OSError("onleesbaar")
            except (OSError, ValueError) as exc:
                log.warning("[%s] AI-resultaat bewaren mislukt: %s", self.name, exc)
        if vals:
            log.info("[%s] waarschijnlijk vals alarm: AI ziet geen %s", self.name,
                     "/".join(sorted((ev.ai_labels or ev.labels) if ev is not None else labels)))
        if self.mqtt and result.get("status") == "ok":
            self.mqtt.publish_ai(self.name, event_id, {**result, "vals": vals})

    def attach_extern(self, result: dict, max_age: float = 300.0) -> bool:
        """Oordeel van de externe AI in HA (vergelijkingsweek) bij het lopende of laatste event."""
        with self._lock:
            ev = self.event or (list(self._pending.values())[-1] if self._pending else None)
            if ev is not None:
                ev.extern = result
                return True
            js = list(self._saved_json.values())[-1] if self._saved_json else None
        if js is None:
            return False
        try:
            start = datetime.fromisoformat(json.loads(js.read_text())["start"])
        except (OSError, ValueError, KeyError):
            return False
        if (datetime.now().astimezone() - start).total_seconds() > max_age + float(self.cfg.max_clip_seconds):
            return False
        return self.store.update(js, {"extern": result}) is not None

    # ------------------------------------------------------------------ vals alarm (gebruiker)
    def _load_false_spots(self) -> list[dict]:
        try:
            return [s for s in json.loads(self._false_file.read_text())
                    if s.get("label") in self.object_active and len(s.get("box", [])) == 4]
        except (FileNotFoundError, ValueError):
            return []

    def _save_false_spots(self) -> None:
        tmp = self._false_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.false_spots, indent=2))
        tmp.replace(self._false_file)

    def _at_false_spot(self, d: Detection, shape: tuple[int, int]) -> bool:
        """Zelfde soort object op (vrijwel) dezelfde plek als een als vals gemarkeerd event."""
        h, w = shape
        box = (d.box[0] / w, d.box[1] / h, d.box[2] / w, d.box[3] / h)
        return any(s["label"] == d.label and (iou(tuple(s["box"]), box) >= FALSE_SPOT_IOU
                                              or contained(box, tuple(s["box"])) >= 0.9)
                   for s in self.false_spots)

    def add_false_spot(self, label: str, box: list[float], event_id: str) -> bool:
        if label not in self.object_active:
            return False
        with self._lock:
            self.false_spots = [s for s in self.false_spots if s.get("event") != event_id]
            self.false_spots.append({"label": label, "box": box, "event": event_id,
                                     "sinds": datetime.now().astimezone().isoformat(timespec="seconds")})
            self.false_spots = self.false_spots[-FALSE_SPOTS_MAX:]
            self._save_false_spots()
        log.info("[%s] vals alarm onthouden: %s op %s", self.name, label, box)
        return True

    def remove_false_spot(self, event_id: str | None = None) -> int:
        """Eén plek (van dat event) of alles vergeten."""
        with self._lock:
            before = len(self.false_spots)
            self.false_spots = [s for s in self.false_spots
                                if event_id is not None and s.get("event") != event_id]
            self._save_false_spots()
            return before - len(self.false_spots)

    # ------------------------------------------------------------------ periodiek
    def _tick_loop(self) -> None:
        while not self._stop.wait(1.0):
            try:
                self._tick()
            except Exception:  # noqa: BLE001
                log.exception("[%s] tick-fout", self.name)

    def _tick(self) -> None:
        now = time.monotonic()
        cooldown = float(self.cfg.motion["cooldown_seconds"])
        with self._lock:
            if self.motion_active and now - self._last_motion > cooldown:
                self.motion_active = False
                self._publish_binary("beweging", False)
            for lbl, on in self.object_active.items():
                if on and now - self._last_object[lbl] > cooldown:
                    self.object_active[lbl] = False
                    self._publish_binary(lbl, False)
            loiter = float(self.cfg.objects["loiter_seconds"])
            if (loiter > 0 and not self.loitering and self.object_active["persoon"]
                    and self._person_since is not None and now - self._person_since >= loiter
                    and self.face_name in ("geen", "onbekend")
                    and not (self.event is not None and self.event.vals)
                    and not self._recent_arrivals(now)):
                self.loitering = True
                log.info("[%s] persoon hangt rond (%.0f s in beeld)", self.name, now - self._person_since)
                self._publish_binary("rondhangen", True)
            if now - self._tracks_saved >= STATIONARY_SAVE_EVERY:
                self._save_tracks(now)
            if self._plate_watch:
                self._check_plate_watch(now)
            ev = self.event
            if ev is not None and (now - ev.last_trigger > self.cfg.post_seconds
                                   or now - ev.start_mono > self.cfg.max_clip_seconds):
                self._end_event()

    def _end_event(self) -> None:
        ev = self.event
        if ev is None:
            return
        self.event = None
        self._pending[ev.id] = ev
        self.recorder.stop_event()
        log.info("[%s] event klaar (%s), clip wordt afgerond", self.name, ev.label)

    # ------------------------------------------------------------------ clip klaar (recorder-thread)
    def _on_clip_done(self, event_id: str, part: Path, duration: float) -> None:
        with self._lock:
            ev = self._pending.pop(event_id, None)
            if ev is None and self.event is not None and self.event.id == event_id:
                # stream viel weg midden in een event: dit deel als clip afronden;
                # komt er daarna weer iets in beeld, dan start er een nieuw event
                ev = self.event
                self.event = None
                self.recorder.stop_event()
        if ev is None or duration < 1.0 or not part.exists():
            log.warning("[%s] lege of onbekende clip %s weggegooid", self.name, event_id)
            part.unlink(missing_ok=True)
            return
        for extra in (f"{ev.id}.meta.json", f"{ev.id}.jpg"):
            (part.parent / extra).unlink(missing_ok=True)
        day_dir, base = clip_paths(self.clips_dir, self.name, ev.start_wall, ev.label)
        mp4 = day_dir / f"{base}.mp4"
        # .parts staat op dezelfde schijf als de clips, dus dit is een snelle rename;
        # shutil.move valt terug op kopiëren als dat ooit niet zo is
        shutil.move(str(part), str(mp4))
        plates = []
        for plate in dict.fromkeys(r.plate for r in ev.plates):   # elk kenteken één keer
            best = max((r for r in ev.plates if r.plate == plate), key=lambda r: r.confidence)
            crop = day_dir / f"{base}.plate{len(plates) + 1}.jpg"
            if best.crop_jpg:
                crop.write_bytes(best.crop_jpg)
            plates.append({"plate": plate, "crop": str(crop.relative_to(self.clips_dir)),
                           "name": self.plates.name_of(plate) if self.plates else None,
                           "confidence": best.confidence})
        recognizer = self.faces
        groups = summarize(ev.faces, recognizer.backend.group if recognizer else 1.1)
        faces = []
        for i, g in enumerate(groups, 1):
            crop = day_dir / f"{base}.face{i}.jpg"
            crop.write_bytes(g.best.crop_jpg)
            faces.append({"crop": str(crop.relative_to(self.clips_dir)), "name": g.name,
                          "score": g.score, "quality": g.best.quality})
        snap_rel = None
        if ev.snapshot_jpg:
            (day_dir / f"{base}.jpg").write_bytes(ev.snapshot_jpg)
            snap_rel = str((day_dir / f"{base}.jpg").relative_to(self.clips_dir))
        end = datetime.now().astimezone()
        record = EventRecord(
            id=ev.id, camera=self.name, label=ev.label,
            labels=sorted(ev.labels, key=lambda lbl: -LABEL_PRIORITY[lbl]),
            score=round(ev.best_score, 3) if ev.best_score is not None else None,
            start=ev.start_wall.isoformat(timespec="seconds"), end=end.isoformat(timespec="seconds"),
            duration=round(duration, 1), clip=str(mp4.relative_to(self.clips_dir)), snapshot=snap_rel,
            faces=faces or None, plates=plates or None, ai=ev.ai,
            zones=sorted(ev.zones) or None,
            vermoedelijk=self._likely_residents(ev, faces),
            box=ev.best_box, vals=ev.vals, extern=ev.extern)
        self.store.add(record)
        self._saved_json[ev.id] = mp4.with_suffix(".json")
        while len(self._saved_json) > 50:
            self._saved_json.pop(next(iter(self._saved_json)))
        log.info("[%s] clip opgeslagen: %s (%.0fs)", self.name, record.clip, duration)
        if self.mqtt:
            self.mqtt.publish_event(self.name, record)
        if recognizer is not None and groups:
            self._learn_faces(recognizer, groups, faces, ev.start_wall)
        self._check_unknown_plates(ev, plates)

    def _recent_arrivals(self, now: float) -> list[str]:
        """Bewoners die volgens HA net thuisgekomen zijn (binnen ARRIVAL_WINDOW)."""
        return sorted(n for n, t in self.arrivals.items() if 0 <= now - t <= ARRIVAL_WINDOW)

    def _likely_residents(self, ev: ActiveEvent, faces: list[dict]) -> list[str] | None:
        """Persoon zonder herkend gezicht: waarschijnlijk een bewoner die net thuiskwam (HA), of
        iemand die vandaag al aan het gezicht herkend is en dezelfde kleding draagt.
        Met een herkend gezicht (en één persoon in beeld) onthouden we juist de kleding."""
        names = [f["name"] for f in faces if f["name"]]
        if "persoon" not in ev.labels:
            return None
        if names:
            if self.bodies is not None and ev.body_emb is not None and ev.max_persons == 1 and len(set(names)) == 1:
                self.bodies.remember(names[0], ev.body_emb)
            return None
        likely = set(self._recent_arrivals(ev.start_mono) or self._recent_arrivals(time.monotonic()))
        if self.bodies is not None and ev.body_emb is not None:
            hit = self.bodies.match(ev.body_emb)
            if hit:
                log.info("[%s] waarschijnlijk %s (kleding/postuur %.2f)", self.name, *hit)
                likely.add(hit[0])
        return sorted(likely) or None

    def _check_unknown_plates(self, ev: ActiveEvent, plates: list[dict]) -> None:
        """Onbekend kenteken: 's nachts meteen melden, overdag pas als de auto blijft staan."""
        if self.plates is None or not self.mqtt:
            return
        for p in plates:
            if p["name"] or self.plates.near_known(p["plate"]):
                continue
            info = {"plate": p["plate"], "crop": p["crop"], "start": ev.start_wall.isoformat(timespec="seconds"),
                    "snapshot_url": f"{self.store.public_url}/clips/{p['crop']}"}
            hour = ev.start_wall.hour
            if hour >= NIGHT_HOURS[0] or hour < NIGHT_HOURS[1]:
                log.info("[%s] onbekende auto %s 's nachts", self.name, p["plate"])
                self.mqtt.publish_alert(self.name, "onbekende_auto", {**info, "reden": "nacht"})
            elif ev.vehicle_box is not None:
                with self._lock:
                    self._plate_watch.append({**info, "box": ev.vehicle_box,
                                              "due": time.monotonic() + PLATE_PARKED_AFTER})

    def _check_plate_watch(self, now: float) -> None:
        """(lock is al genomen) Onbekende auto's waarvan het kwartier om is: op de achtergrond
        in het huidige beeld kijken of er op die plek nog een auto staat."""
        due = [w for w in self._plate_watch if w["due"] <= now]
        if due:
            self._plate_watch = [w for w in self._plate_watch if w["due"] > now]
            threading.Thread(target=self._check_parked, args=(due,),
                             name=f"parked-{self.name}", daemon=True).start()

    def _check_parked(self, due: list[dict]) -> None:
        try:
            bgr = self.latest_bgr
            if bgr is None or self.detector is None:
                return
            cars = self.detector.detect(bgr.copy(), ["voertuig"], float(self.cfg.objects["min_score"]))
            for w in due:
                if any(iou(c.box, w["box"]) >= 0.3 or contained(c.box, w["box"]) >= 0.8
                       or contained(w["box"], c.box) >= 0.8 for c in cars) and self.mqtt:
                    log.info("[%s] onbekende auto %s staat er al een kwartier", self.name, w["plate"])
                    self.mqtt.publish_alert(self.name, "onbekende_auto",
                                            {k: v for k, v in w.items() if k not in ("box", "due")}
                                            | {"reden": "geparkeerd", "minuten": int(PLATE_PARKED_AFTER // 60)})
        except Exception:  # noqa: BLE001
            log.exception("[%s] controle geparkeerde auto mislukt", self.name)

    def _learn_faces(self, recognizer: FaceRecognizer, groups: list, faces: list[dict],
                     when: datetime) -> None:
        """Na een event: zeker herkende gezichten bijleren, onbekende verzamelen."""
        fcfg = self.cfg.settings["faces"]
        for g, f in zip(groups, faces):
            try:
                if g.name and fcfg.get("auto_learn", True):
                    recognizer.learn_auto(g)
                elif not g.name and fcfg.get("collect_unknown", True):
                    sid = recognizer.unknown.add(g.best, self.name, when, f["crop"])
                    group = recognizer.unknown.group_of(sid) if sid else None
                    # vragen wie het is: bij de eerste keer en nog eens als iemand vaker komt
                    if group and group["count"] in ASK_FACE_AT and self.mqtt:
                        self.mqtt.publish_alert(self.name, "onbekend_gezicht", {
                            "sample": sid, "aantal": group["count"], "suggestie": group["suggest"],
                            "namen": sorted(recognizer.people())[:8]})
            except Exception:  # noqa: BLE001 - leren mag het opslaan nooit hinderen
                log.exception("[%s] gezicht bijleren mislukt", self.name)

    # ------------------------------------------------------------------ MQTT-hulpjes
    def _on_main_online(self, _ok: bool) -> None:
        self._publish_online()

    @property
    def online(self) -> bool:
        return self.sub_online and self.recorder.online

    def _publish_online(self) -> None:
        if self.mqtt:
            self.mqtt.publish_binary(self.name, "online", self.online)

    def _publish_binary(self, which: str, on: bool) -> None:
        if self.mqtt:
            self.mqtt.publish_binary(self.name, which, on)
        if which == "voertuig" and not on and self.plate != "geen":
            self._set_plate("geen", None)
        if which == "persoon" and not on:
            self._person_since = None
            if self.face_name != "geen":
                self._set_face("geen")
            if self.loitering:
                self.loitering = False
                self._publish_binary("rondhangen", False)

    def status(self) -> dict:
        return {
            "name": self.name,
            "online": self.online,
            "main_online": self.recorder.online,
            "sub_online": self.sub_online,
            "codec": self.recorder.codec,
            "main_bitrate_kbps": round(self.recorder.bitrate_kbps),
            "buffer_seconds": round(self.recorder.ring.seconds, 1),
            "buffer_mb": round(self.recorder.ring.nbytes / 1e6, 2),
            "detect_enabled": self.detect_enabled,
            "record_enabled": self.record_enabled,
            "motion": self.motion_active,
            "objects": dict(self.object_active),
            "event_active": self.event is not None,
            "detector_device": getattr(self.detector, "device", None),
            "stationary": [{"label": t.label, "seconds": round(time.monotonic() - t.first_seen)}
                           for t in self._tracks],
            "max_infer_fps": self.max_infer_fps,
            "faces": self.faces is not None,
            "face": self.face_name,
            "plates": self.plates is not None,
            "plate": self.plate,
            "always_detect": self.always_detect,
            "loitering": self.loitering,
            **self.stats,
        }
