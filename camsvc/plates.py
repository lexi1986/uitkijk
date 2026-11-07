"""Licence plate recognition, fully local: a YOLOv9 detector finds the plate, an OCR model
reads the characters (fast-alpr, ONNX Runtime on the CPU). Dutch plates only: the text read
must fit one of the Dutch sidecodes, with correction of typical misreads (O/0, B/8, S/5, ...).

Known plates with a name are stored in <state_dir>/plates.json.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

DETECTOR_MODEL = "yolo-v9-t-384-license-plate-end2end"
OCR_MODEL = "european-plates-mobile-vit-v2-model"
MIN_PLATE_CONF = 0.5       # detector: this sure that it is a licence plate
MIN_OCR_CONF = 0.6         # OCR: average confidence per character
DETECT_MAX_SIDE = 1280     # vehicle crop at most this large for the detector

# Dutch sidecodes 1-14: L = letter, D = digit, with the dashes as on the plate.
SIDECODES = ["LL-DD-DD", "DD-DD-LL", "DD-LL-DD", "LL-DD-LL", "LL-LL-DD", "DD-LL-LL",
             "DD-LLL-D", "D-LLL-DD", "LL-DDD-L", "L-DDD-LL", "LLL-DD-L", "L-DD-LLL",
             "D-LL-DDD", "DDD-LL-D"]
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8",
            "G": "6", "T": "7"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T", "4": "A"}


def normalize_nl(text: str) -> str | None:
    """Text read -> Dutch plate with dashes (e.g. 'GKB31R' -> 'GKB-31-R'), or None.

    Tries every sidecode; per sidecode at most 1 character is 'corrected' (O->0 in a digit
    position, 8->B in a letter position, ...). The sidecode with the fewest corrections wins.
    """
    raw = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(raw) != 6:
        return None
    best: tuple[int, str] | None = None
    for code in SIDECODES:
        kinds = code.replace("-", "")
        out, fixes = [], 0
        for ch, kind in zip(raw, kinds):
            if kind == "D" and not ch.isdigit():
                ch, fixes = TO_DIGIT.get(ch, ""), fixes + 1
            elif kind == "L" and not ch.isalpha():
                ch, fixes = TO_LETTER.get(ch, ""), fixes + 1
            if not ch:
                break
            out.append(ch)
        else:
            if fixes <= 1 and (best is None or fixes < best[0]):
                it = iter(out)
                best = (fixes, "".join(c if c == "-" else next(it) for c in code))
    return best[1] if best else None


MAX_MISREAD = 2   # a known plate when at most this many characters were read differently


def closest_known(plate: str, known) -> str | None:
    """Closest known plate, with at most MAX_MISREAD misread characters.
    shortcut: a visitor with an almost identical plate counts as the known car; rare."""
    flat = plate.replace("-", "")
    diffs = [(sum(a != b for a, b in zip(k.replace("-", ""), flat)), k) for k in known
             if len(k.replace("-", "")) == len(flat)]
    best = min(diffs, default=None)
    return best[1] if best and best[0] <= MAX_MISREAD else None


def valid_plate(plate: str) -> bool:
    return normalize_nl(plate) == plate



def plates_in_text(text: str) -> list[str]:
    """Plates mentioned in a text ('car with plate GKB 31 R' -> ['GKB-31-R']). Only exact Dutch
    plates with letters and digits, without 'corrected' characters: ordinary words drop out."""
    found = []
    words = re.findall(r"[A-Z0-9]+", (text or "").upper())
    for i in range(len(words)):      # 1 to 3 adjacent words ('GKB 31 R', 'GKB-31-R', 'GKB31R')
        for n in (1, 2, 3):
            raw = "".join(words[i:i + n])
            plate = normalize_nl(raw) if len(raw) == 6 else None
            if (plate and plate.replace("-", "") == raw and any(c.isdigit() for c in raw)
                    and any(c.isalpha() for c in raw) and plate not in found):
                found.append(plate)
    return found

@dataclass
class PlateRead:
    plate: str            # Dutch plate with dashes
    confidence: float     # average OCR confidence
    crop_jpg: bytes       # crop of the plate


class PlateReader:
    def __init__(self, state_dir: Path):
        self._file = state_dir / "plates.json"
        self._lock = threading.Lock()
        self._alpr = None
        self.known: dict[str, str] = self._load()

    # ------------------------------------------------------------------ known plates
    def _load(self) -> dict[str, str]:
        try:
            data = json.loads(self._file.read_text())
            return {p: str(n) for p, n in data.items() if valid_plate(p)}
        except (FileNotFoundError, ValueError):
            return {}

    def _save(self) -> None:
        tmp = self._file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.known, indent=2, ensure_ascii=False))
        tmp.replace(self._file)

    def set_name(self, plate: str, name: str) -> None:
        plate = normalize_nl(plate) or ""
        name = name.strip()
        if not plate:
            raise ValueError("Not a valid Dutch licence plate")
        if not name or len(name) > 40:
            raise ValueError("A name is required (max. 40 characters)")
        with self._lock:
            self.known[plate] = name
            self._save()

    def forget(self, plate: str) -> bool:
        with self._lock:
            if self.known.pop(plate, None) is None:
                return False
            self._save()
            return True

    def name_of(self, plate: str) -> str | None:
        return self.known.get(plate)

    def known_plate(self, plate: str) -> str | None:
        """Known plate within MAX_MISREAD characters (typical misread, GKB/GKS)."""
        return closest_known(plate, self.known)

    def near_known(self, plate: str) -> str | None:
        """Name of the known plate within MAX_MISREAD characters."""
        known = self.known_plate(plate)
        return self.known[known] if known else None

    # ------------------------------------------------------------------ reading
    def _model(self):
        # load on first use: the models (~15 MB) are then downloaded once to ~/.cache
        if self._alpr is None:
            import onnxruntime as ort
            from fast_alpr import ALPR

            def opts():
                # 1 thread: light, and no affinity errors in an LXC; only log real errors
                o = ort.SessionOptions()
                o.intra_op_num_threads = 1
                o.inter_op_num_threads = 1
                o.log_severity_level = 3
                return o

            self._alpr = ALPR(detector_model=DETECTOR_MODEL, detector_conf_thresh=MIN_PLATE_CONF,
                              detector_providers=["CPUExecutionProvider"], detector_sess_options=opts(),
                              ocr_model=OCR_MODEL, ocr_device="cpu",
                              ocr_providers=["CPUExecutionProvider"], ocr_sess_options=opts())
            log.info("Plate models loaded (%s, %s)", DETECTOR_MODEL, OCR_MODEL)
        return self._alpr

    def warm_up(self) -> None:
        """Load the models in advance (in a background thread at startup)."""
        try:
            with self._lock:
                self._model()
        except Exception as exc:  # noqa: BLE001
            log.error("Loading plate models failed: %s", exc)

    def read(self, bgr: np.ndarray) -> list[PlateRead]:
        """All Dutch licence plates in a (cropped) image."""
        h, w = bgr.shape[:2]
        scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
        img = cv2.resize(bgr, (int(w * scale), int(h * scale))) if scale < 1 else bgr
        with self._lock:
            results = self._model().predict(img)
        out = []
        for r in results:
            if r.ocr is None or not r.ocr.text:
                continue
            conf = r.ocr.confidence
            conf = float(np.mean(conf)) if isinstance(conf, (list, tuple, np.ndarray)) else float(conf)
            plate = normalize_nl(r.ocr.text)
            if plate is None or conf < MIN_OCR_CONF:
                log.debug("Plate ignored: %r (%.2f)", r.ocr.text, conf)
                continue
            b = r.detection.bounding_box
            pad = 6
            crop = img[max(0, b.y1 - pad):b.y2 + pad, max(0, b.x1 - pad):b.x2 + pad]
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            out.append(PlateRead(plate, round(conf, 3), buf.tobytes() if ok else b""))
        return out


def vote(reads: list[PlateRead]) -> PlateRead | None:
    """Most frequently read plate across frames (on a tie, the most confident reading)."""
    if not reads:
        return None
    counts = Counter(r.plate for r in reads)
    top = max(counts.values())
    return max((r for r in reads if counts[r.plate] == top), key=lambda r: r.confidence)
