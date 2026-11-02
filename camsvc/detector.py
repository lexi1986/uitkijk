"""Objectdetectie met een YOLO-model.

Standaard ONNX Runtime op de CPU (.onnx); optioneel een Google Coral (Edge TPU, .tflite)
of OpenVINO (.xml) voor de iGPU. Werkt met YOLOv8/YOLO11-exports. De uitvoer heeft de
vorm [1, 4 + klassen, N]. Eén gedeelde detector voor alle camera's.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import LABEL_CLASSES

log = logging.getLogger(__name__)

CLASS_TO_LABEL = {cls: label for label, classes in LABEL_CLASSES.items() for cls in classes}


@dataclass(slots=True)
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]   # x1, y1, x2, y2 in pixels van het invoerframe


_OV_PROBE = r"""
import sys, numpy as np, openvino as ov
core = ov.Core()
dev, s = sys.argv[2], int(sys.argv[3])
if dev == "GPU" and not any(d.startswith("GPU") for d in core.available_devices):
    sys.exit(3)
m = core.read_model(sys.argv[1])
m.reshape([1, 3, s, s])
c = core.compile_model(m, dev, {"PERFORMANCE_HINT": "LATENCY"})
c.create_infer_request().infer({0: np.zeros((1, 3, s, s), np.float32)})
"""


def openvino_works(model_path: str | Path, device: str, input_size: int) -> bool:
    """Test OpenVINO op een apparaat in een apart proces. Een kapotte driver of een
    OpenVINO-versie die niet met de CPU/container overweg kan, kan crashen (segfault);
    dat mag de service zelf niet overkomen."""
    try:
        r = subprocess.run([sys.executable, "-c", _OV_PROBE, str(model_path), device, str(input_size)],
                           capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        log.warning("OpenVINO-test op %s duurde te lang", device)
        return False
    if r.returncode == 0:
        return True
    if r.returncode == 3:
        log.info("Geen GPU gevonden voor OpenVINO")
    else:
        detail = (r.stderr.strip().splitlines() or [f"exitcode {r.returncode}"])[-1]
        log.warning("OpenVINO op %s niet bruikbaar (%s)", device, detail)
    return False


class ObjectDetector:
    """Laadt een YOLO-model: .onnx via ONNX Runtime (CPU), .tflite op een Coral (Edge TPU),
    .xml via OpenVINO (GPU/CPU)."""

    def __init__(self, model_path: str | Path, device: str = "AUTO", input_size: int = 320):
        self.input_size = int(input_size)
        self._lock = threading.Lock()
        self.accelerated = False   # True = model draait niet op de CPU
        path = Path(model_path)
        if path.suffix == ".onnx":
            self._load_onnx(path)
        elif path.suffix == ".tflite":
            self._load_edgetpu(path)
        else:
            self._load_openvino(path, device.upper())
        log.info("Objectmodel %s geladen via %s op %s (invoer %dpx)",
                 path.name, self.backend, self.device, self.input_size)

    def _load_onnx(self, path: Path) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        if isinstance(inp.shape[2], int):
            self.input_size = int(inp.shape[2])   # vaste grootte uit de export
        self.backend, self.device = "ONNX Runtime", "CPU"
        self._run = lambda blob: self.session.run(None, {self.input_name: blob})[0]

    def _load_edgetpu(self, path: Path) -> None:
        """Model voor de Coral (Ultralytics `format=edgetpu`, int8). Zonder Coral een fout."""
        try:
            from ai_edge_litert.interpreter import Interpreter, load_delegate
        except ImportError:
            from tflite_runtime.interpreter import Interpreter, load_delegate
        delegate = load_delegate("libedgetpu.so.1")   # ValueError als er geen Coral is
        interp = Interpreter(model_path=str(path), experimental_delegates=[delegate])
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        self.input_size = size = int(inp["shape"][1])   # NHWC
        in_scale, in_zero = inp["quantization"]
        out_scale, out_zero = out["quantization"]
        in_dtype = inp["dtype"]
        in_range = np.iinfo(in_dtype) if np.issubdtype(in_dtype, np.integer) else None

        def run(blob: np.ndarray) -> np.ndarray:
            x = blob.transpose(0, 2, 3, 1)               # NCHW → NHWC
            if in_scale:
                x = x / in_scale + in_zero
            if in_range is not None:
                x = np.clip(np.round(x), in_range.min, in_range.max)
            interp.set_tensor(inp["index"], x.astype(in_dtype))
            interp.invoke()
            y = interp.get_tensor(out["index"]).astype(np.float32)
            if out_scale:
                y = (y - out_zero) * out_scale
            y[:, :4] *= size     # TFLite-exports geven kaders genormaliseerd (0-1)
            return y

        self.backend, self.device = "TFLite", "Edge TPU"
        self.accelerated = True
        self._run = run

    def _load_openvino(self, path: Path, wanted: str) -> None:
        candidates = {"AUTO": ["GPU", "CPU"], "GPU": ["GPU", "CPU"]}.get(wanted, [wanted])
        chosen = next((d for d in candidates if openvino_works(path, d, self.input_size)), None)
        if chosen is None:
            raise RuntimeError("OpenVINO werkt op geen enkel apparaat in deze container; "
                               "gebruik een .onnx-model (zie README)")
        import openvino as ov

        core = ov.Core()
        model = core.read_model(str(path))
        model.reshape([1, 3, self.input_size, self.input_size])
        compiled = core.compile_model(model, chosen, {"PERFORMANCE_HINT": "LATENCY"})
        request = compiled.create_infer_request()
        self.backend, self.device = "OpenVINO", chosen
        self.accelerated = chosen == "GPU"
        self._run = lambda blob: next(iter(request.infer({0: blob}).values()))

    def _letterbox(self, bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = bgr.shape[:2]
        s = self.input_size
        scale = min(s / w, s / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        px, py = (s - nw) // 2, (s - nh) // 2
        canvas[py:py + nh, px:px + nw] = resized
        return canvas, scale, px, py

    def detect(self, bgr: np.ndarray, labels: list[str], min_score: float) -> list[Detection]:
        if not labels:
            return []
        wanted = {cls for lbl in labels for cls in LABEL_CLASSES[lbl]}
        img, scale, px, py = self._letterbox(bgr)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, swapRB=True)  # NCHW, RGB, 0-1
        with self._lock:
            pred = self._run(blob)[0]                # [4 + C, N]
        pred = pred.T                                # [N, 4 + C]
        class_scores = pred[:, 4:]
        cls_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(len(cls_ids)), cls_ids]
        keep = (scores >= min_score) & np.isin(cls_ids, list(wanted))
        if not keep.any():
            return []
        boxes_cxcywh = pred[keep, :4]
        scores, cls_ids = scores[keep], cls_ids[keep]
        h, w = bgr.shape[:2]
        x1 = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2 - px) / scale
        y1 = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2 - py) / scale
        bw = boxes_cxcywh[:, 2] / scale
        bh = boxes_cxcywh[:, 3] / scale
        rects = [[float(a), float(b), float(c), float(d)] for a, b, c, d in zip(x1, y1, bw, bh)]
        idx = cv2.dnn.NMSBoxes(rects, scores.astype(float).tolist(), min_score, 0.45)
        result = []
        for i in np.array(idx).flatten():
            x, y, bw_, bh_ = rects[i]
            box = (max(int(x), 0), max(int(y), 0), min(int(x + bw_), w - 1), min(int(y + bh_), h - 1))
            result.append(Detection(CLASS_TO_LABEL[int(cls_ids[i])], float(scores[i]), box))
        return result


def draw_detections(bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
        text = f"{d.label} {d.score:.0%}"
        cv2.putText(out, text, (x1 + 3, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 200, 255), 1, cv2.LINE_AA)
    return out
