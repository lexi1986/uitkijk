"""Herkennen aan kleding en postuur (persoon-her-identificatie), volledig lokaal.

Model: YouTu-ReID uit OpenCV Zoo (Apache-2.0), via onnxruntime. Is iemand ergens aan het
gezicht herkend, dan onthouden we die dag ook hoe die persoon eruitziet (kleding, postuur).
Een persoon zonder herkend gezicht, bijv. van achteren bij het hek, wordt daarmee vergeleken.
Kleding wisselt per dag, dus alleen beelden van de laatste KEEP_HOURS tellen.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODEL_FILE = "person_reid_youtu_2021nov.onnx"
MATCH = 0.5          # zelfde persoon, zelfde kleding: 0.5-0.85; andere personen bleven onder 0.25
MARGIN = 0.15        # en duidelijk beter dan de op één na beste persoon
KEEP_HOURS = 12.0
PER_PERSON = 20
MIN_CROP = (40, 90)  # kleinere personen (breedte, hoogte in px) zeggen te weinig
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class BodyMatcher:
    def __init__(self, models_dir: Path, state_file: Path):
        import onnxruntime as ort
        ort.disable_telemetry_events()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.log_severity_level = 3
        opts.enable_cpu_mem_arena = False
        self._session = ort.InferenceSession(str(models_dir / MODEL_FILE), opts,
                                             providers=["CPUExecutionProvider"])
        self._lock = threading.Lock()
        self._file = state_file
        self._seen: dict[str, list[tuple[float, np.ndarray]]] = {}
        self._load()

    @staticmethod
    def available(models_dir: Path) -> bool:
        return (models_dir / MODEL_FILE).exists()

    def embed(self, crop: np.ndarray) -> np.ndarray | None:
        """Kenmerkvector van een uitsnede van één persoon (BGR), of None als die te klein is."""
        h, w = crop.shape[:2]
        if w < MIN_CROP[0] or h < MIN_CROP[1]:
            return None
        x = cv2.resize(crop, (128, 256))[:, :, ::-1].astype(np.float32) / 255.0
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        with self._lock:
            v = self._session.run(None, {self._session.get_inputs()[0].name: x})[0].reshape(-1)
        return v / max(float(np.linalg.norm(v)), 1e-6)

    # ------------------------------------------------------------------ onthouden
    def _load(self) -> None:
        try:
            data = json.loads(self._file.read_text())
        except (FileNotFoundError, ValueError):
            return
        for name, items in data.items():
            self._seen[name] = [(t, np.asarray(e, np.float32)) for t, e in items]
        self._prune()

    def _save(self) -> None:
        data = {n: [[t, [round(float(x), 5) for x in e]] for t, e in items] for n, items in self._seen.items()}
        tmp = self._file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self._file)

    def _prune(self) -> None:
        cutoff = time.time() - KEEP_HOURS * 3600
        self._seen = {n: [(t, e) for t, e in items if t >= cutoff][-PER_PERSON:]
                      for n, items in self._seen.items()}
        self._seen = {n: items for n, items in self._seen.items() if items}

    def remember(self, name: str, emb: np.ndarray) -> None:
        with self._lock:
            self._seen.setdefault(name, []).append((time.time(), emb))
            self._prune()
            self._save()

    def match(self, emb: np.ndarray) -> tuple[str, float] | None:
        """Wie van vandaag lijkt qua kleding en postuur het meest? Alleen bij een duidelijke match."""
        with self._lock:
            self._prune()
            per = {n: max(float(e @ emb) for _, e in items) for n, items in self._seen.items()}
        if not per:
            return None
        ranked = sorted(per.items(), key=lambda kv: -kv[1])
        name, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if best >= MATCH and best - second >= MARGIN:
            return name, round(best, 3)
        return None
