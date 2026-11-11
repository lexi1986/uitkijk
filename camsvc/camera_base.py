"""Shared constants, helpers and data classes of the camera modules."""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import av
import cv2
import numpy as np

from .config import CameraConfig
from .detector import Detection, ObjectDetector, draw_detections
from .events import NIGHT_HOURS, EventRecord, EventStore, clip_paths, unknown_person
from .faces import FaceGroup, FaceMatch, FaceRecognizer, confirmed_weak, summarize
from .motion import MotionDetector, build_mask, motion_region, point_masked, zones_at
from .plates import VISITOR_DAYS, PlateRead, PlateReader, plates_in_text, vote
from .recorder import RTSP_OPTIONS, MainRecorder
from .tracker import ObjectTrack, Tracker, plausible_shape

if TYPE_CHECKING:
    from .ai_describe import AiDescriber
    from .mqtt import MqttBridge

log = logging.getLogger("camsvc.camera")   # one logger name for all camera modules

ALWAYS_IDLE_FPS = 2.0     # always: without motion (2 hits within 3 s still confirm)
TRACKED_LABELS = ("person", "vehicle")
TRACK_MIN_MEDIAN = 0.6    # measured on real clips: keeps real people, drops 3 of 20 false ones; vehicles gain nothing
LABEL_PRIORITY = {"doorbell": 4, "person": 3, "vehicle": 2, "animal": 1, "motion": 0}
# Labels that can stand still without being interesting (parked car, sleeping cat).
STATIONARY_LABELS = {"vehicle", "animal"}
STATIONARY_IOU = 0.7      # this much overlap with the previous spot = not moved
# Only forget an object after not seeing it at all for this long. Do not count how often the
# model misses it: at night the confidence hovers around the threshold and a parked car would
# keep becoming 'new'. A car that leaves or arrives moves and counts normally.
STATIONARY_FORGET = 3600.0
FULL_FRAME_EVERY = 10.0   # look at the whole frame at least this often, even when zooming is possible
STATIONARY_MAX_TRACKS = 30
STATIONARY_SAVE_EVERY = 30.0   # seconds between saving stationary objects
# Part of a stationary object only counts as that object when it was seen recently; a car that
# left and comes back later triggers an event again.
STATIONARY_PART_RECENT = 900.0
STATIONARY_REFRESH = 300.0    # this often, when idle, check whether stationary objects are still there
# Crop for the AI description: this many times the box, at least this share of the frame height.
AI_CROP_MARGIN = 2.0
AI_CROP_MIN = 0.45
# "False alarm" from the web page: the same kind of object in almost the same spot no longer counts.
FALSE_SPOT_IOU = 0.6
FALSE_SPOTS_MAX = 50
# Doubtful detections (score below objects.verify_below) are first verified on a sharp, zoomed
# crop from the main stream. A drainpipe or garden bag scores 0 there, a real person 0.8+.
VERIFY_LABELS = {"person", "animal"}
PLATE_PARKED_AFTER = 900.0    # unknown vehicle that stays this long: alert
RING_FACE_ATTEMPTS = 3        # on a doorbell ring, search the whole frame for a face this often (1 s apart)
ASK_FACE_AT = (1, 3)          # ask who it is the first time and again when an unknown face returns
RESCAN_STEP = 0.5             # saved clip: look for faces every half second ...
RESCAN_MAX_SECONDS = 120      # ... in at most the first two minutes
RESCAN_MIN_FRAMES = 2         # a newly found face must appear in this many frames
RESCAN_BODY_STEP = 1.0        # clothing: look for the largest person every second
UNKNOWN_PERSON_DELAY = 30.0   # after rescanning the clip, wait this long for the AI verdict
_RESCAN_SLOT = threading.Semaphore(1)   # one clip at a time, across all cameras
_RESCAN_WAITING = threading.Semaphore(4)   # at most this many clips waiting; skip the rest
BODY_MIN_FRAC = 0.25          # clothing only from a person filling this share of the frame height: far away
                              # on the street (0.13-0.24) a passer-by looked like a resident (0.6-0.7)
BODY_ATTEMPTS = 3             # this many sharp crops of the person per event
BODY_INTERVAL = 2.0
ARRIVAL_WINDOW = 300.0        # this long after arriving home (per HA) an unknown person is probably that resident
VERIFY_MIN = 0.5          # the sharp crop must score at least this
# At night (infrared, black and white) a real person scores much lower, also on the sharp frame:
# someone stepping out of a car stayed around 0.3-0.45 and the check gave 0.32
NIGHT_PERSON_MIN = 0.4    # check people from this score up (instead of min_score)
VERIFY_MIN_NIGHT = 0.25
VERIFY_KEEP_OK = 30.0     # confirmed: do not verify again for this long (while it stays nearby)
VERIFY_KEEP_BAD = 300.0   # ignore a rejected spot for this long


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def contained(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Which share of box a lies inside box b (0-1)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


@dataclass
class Verdict:
    """Outcome of verifying a doubtful detection at a spot in the frame."""
    label: str
    box: tuple[int, int, int, int]
    until: float
    ok: bool


def zoom_score(main: np.ndarray, box: tuple[int, int, int, int], sub_shape: tuple[int, int],
               detector: ObjectDetector, label: str) -> float:
    """Highest score for `label` in a square crop of the main image around a box from the sub
    stream (at least 320 px, so the model does not have to upscale)."""
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
    """An object that stays in (almost) the same spot."""
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
    vehicle_box: tuple[int, int, int, int] | None = None   # last position of the vehicle
    body_emb: np.ndarray | None = None    # clothing/build of the largest person (re-identification)
    body_area: int = 0
    body_attempts: int = 0
    face_crops: dict[int, str] = field(default_factory=dict)   # id(face frame) -> saved crop
    objects: dict[int, ObjectTrack] = field(default_factory=dict)   # tracked people/vehicles
    last_body_try: float = 0.0
    max_persons: int = 0
    loitering: bool = False
    last_plate_try: float = 0.0
    best_box: list[float] | None = None   # best detection as a fraction of the frame
    ai_labels: set[str] = field(default_factory=set)   # labels at the time the AI description was requested
    false_alarm: str | None = None
    external: dict | None = None
