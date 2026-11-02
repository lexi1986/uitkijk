"""Prepare models at startup (once, with checksum verification).

- Object detection (YOLO11n as ONNX): part of the Docker image and copied to the model folder.
- Faces and clothing (only when a camera has faces.enabled): YuNet and SFace (OpenCV Zoo),
  YouTu-ReID (Apache-2.0) and ArcFace w600k_r50 from InsightFace buffalo_l. That last model is for
  non-commercial use only, so it is never shipped; the user's own installation downloads it here.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger(__name__)

ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
FACE_MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "face_recognition_sface_2021dec.onnx": (
        f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
    "person_reid_youtu_2021nov.onnx": (
        f"{ZOO}/person_reid_youtureid/person_reid_youtu_2021nov.onnx",
        "0579683334d4b9440221606dcb461656dd0dc64143b18f48faedaced9b4f580d"),
}
ARCFACE = ("w600k_r50.onnx", "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
           "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43")
BUNDLED_DIR = Path(os.environ.get("CAMSVC_BUNDLED_MODELS", "/opt/camsvc-models"))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ok(path: Path, sha: str) -> bool:
    return path.exists() and _sha(path) == sha


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def _fetch(dest: Path, url: str, sha: str) -> None:
    if _ok(dest, sha):
        return
    log.info("Downloading model %s…", dest.name)
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(_download(url))
    if _sha(tmp) != sha:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"checksum of {dest.name} does not match")
    tmp.replace(dest)


def _fetch_arcface(dest_dir: Path) -> None:
    name, url, sha = ARCFACE
    dest = dest_dir / name
    if _ok(dest, sha):
        return
    log.info("Downloading model %s (zip of ~290 MB, non-commercial use only)…", name)
    data = zipfile.ZipFile(io.BytesIO(_download(url))).read(name)
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError(f"checksum of {name} does not match")
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)


def ensure_models(cfg: "Config") -> None:
    """Prepare missing models. Log errors, never block startup: without face models everything
    else runs normally, without the object model only motion detection runs."""
    model = Path(cfg.model["path"])
    try:
        model.parent.mkdir(parents=True, exist_ok=True)
        bundled = BUNDLED_DIR / model.name
        if not model.exists() and bundled.exists():
            shutil.copy2(bundled, model)
            log.info("Object model %s prepared", model.name)
    except OSError as exc:
        log.error("Model folder %s not usable: %s", model.parent, exc)
        return
    if not any(c.settings["faces"]["enabled"] for c in cfg.cameras):
        return
    try:
        for name, (url, sha) in FACE_MODELS.items():
            _fetch(model.parent / name, url, sha)
        _fetch_arcface(model.parent)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        log.error("Downloading face models failed (%s); will retry at the next start", exc)
