"""Kentekenherkenning, volledig lokaal: een YOLOv9-detector zoekt de plaat, een OCR-model
leest de tekens (fast-alpr, ONNX Runtime op de CPU). Alleen Nederlandse kentekens: de
gelezen tekst moet in een van de NL-sidecodes passen, met correctie van typische
leesfouten (O/0, B/8, S/5, ...).

Bekende kentekens met een naam staan in <state_dir>/plates.json.
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
MIN_PLATE_CONF = 0.5       # detector: zo zeker moet het een kentekenplaat zijn
MIN_OCR_CONF = 0.6         # OCR: gemiddelde zekerheid per teken
DETECT_MAX_SIDE = 1280     # uitsnede van het voertuig hooguit zo groot voor de detector

# NL-sidecodes 1-14: L = letter, D = cijfer, met de streepjes zoals op de plaat.
SIDECODES = ["LL-DD-DD", "DD-DD-LL", "DD-LL-DD", "LL-DD-LL", "LL-LL-DD", "DD-LL-LL",
             "DD-LLL-D", "D-LLL-DD", "LL-DDD-L", "L-DDD-LL", "LLL-DD-L", "L-DD-LLL",
             "D-LL-DDD", "DDD-LL-D"]
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8",
            "G": "6", "T": "7"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T", "4": "A"}


def normalize_nl(text: str) -> str | None:
    """Gelezen tekst -> NL-kenteken met streepjes (bijv. 'GKB31R' -> 'GKB-31-R'), of None.

    Probeert elke sidecode; per sidecode wordt hooguit 1 teken 'verbeterd' (O->0 op een
    cijferplek, 8->B op een letterplek, ...). De sidecode met de minste correcties wint.
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


def valid_plate(plate: str) -> bool:
    return normalize_nl(plate) == plate


@dataclass
class PlateRead:
    plate: str            # NL-kenteken met streepjes
    confidence: float     # gemiddelde OCR-zekerheid
    crop_jpg: bytes       # uitsnede van de plaat


class PlateReader:
    def __init__(self, state_dir: Path):
        self._file = state_dir / "plates.json"
        self._lock = threading.Lock()
        self._alpr = None
        self.known: dict[str, str] = self._load()

    # ------------------------------------------------------------------ bekende kentekens
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
            raise ValueError("Geen geldig Nederlands kenteken")
        if not name or len(name) > 40:
            raise ValueError("Naam is verplicht (max. 40 tekens)")
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

    def near_known(self, plate: str) -> str | None:
        """Bekend kenteken dat hooguit één teken verschilt (typische leesfout, GKB/GKS)."""
        flat = plate.replace("-", "")
        for known, name in self.known.items():
            k = known.replace("-", "")
            if len(k) == len(flat) and sum(a != b for a, b in zip(k, flat)) <= 1:
                return name
        return None

    # ------------------------------------------------------------------ lezen
    def _model(self):
        # pas bij het eerste gebruik laden: de modellen (± 15 MB) worden dan zo nodig
        # eenmalig gedownload naar ~/.cache
        if self._alpr is None:
            import onnxruntime as ort
            from fast_alpr import ALPR

            def opts():
                # 1 thread: licht, en geen affiniteitsfouten in een LXC; alleen echte fouten loggen
                o = ort.SessionOptions()
                o.intra_op_num_threads = 1
                o.inter_op_num_threads = 1
                o.log_severity_level = 3
                return o

            self._alpr = ALPR(detector_model=DETECTOR_MODEL, detector_conf_thresh=MIN_PLATE_CONF,
                              detector_providers=["CPUExecutionProvider"], detector_sess_options=opts(),
                              ocr_model=OCR_MODEL, ocr_device="cpu",
                              ocr_providers=["CPUExecutionProvider"], ocr_sess_options=opts())
            log.info("Kentekenmodellen geladen (%s, %s)", DETECTOR_MODEL, OCR_MODEL)
        return self._alpr

    def warm_up(self) -> None:
        """Modellen alvast laden (in een achtergrondthread bij het opstarten)."""
        try:
            with self._lock:
                self._model()
        except Exception as exc:  # noqa: BLE001
            log.error("Kentekenmodellen laden mislukt: %s", exc)

    def read(self, bgr: np.ndarray) -> list[PlateRead]:
        """Alle Nederlandse kentekens in een (uitgesneden) beeld."""
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
                log.debug("Kenteken genegeerd: %r (%.2f)", r.ocr.text, conf)
                continue
            b = r.detection.bounding_box
            pad = 6
            crop = img[max(0, b.y1 - pad):b.y2 + pad, max(0, b.x1 - pad):b.x2 + pad]
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            out.append(PlateRead(plate, round(conf, 3), buf.tobytes() if ok else b""))
        return out


def vote(reads: list[PlateRead]) -> PlateRead | None:
    """Meest gelezen kenteken over meerdere beelden (bij gelijkspel de zekerste lezing)."""
    if not reads:
        return None
    counts = Counter(r.plate for r in reads)
    top = max(counts.values())
    return max((r for r in reads if counts[r.plate] == top), key=lambda r: r.confidence)
