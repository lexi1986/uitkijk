"""Describe old recordings again with the current AI prompt (PROMPT_VERSION).

One-off, preferably at night: one frame at a time, from the clip itself (the snapshot has
detection boxes). Only events with a successful description of an older version.

  sudo -u camsvc /opt/camera-service/.venv/bin/python /opt/camera-service/deploy/redescribe.py \
      --config /etc/camera-service/config.yaml [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import av

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from camsvc.ai_describe import (PROMPT_VERSION, AiError, OllamaClient, crop_around,  # noqa: E402
                                false_alarm, prepare_image)
from camsvc.camera import AI_CROP_MARGIN, AI_CROP_MIN  # noqa: E402
from camsvc.config import load_config  # noqa: E402

log = logging.getLogger("redescribe")


def frame_at(clip: Path, seconds: float):
    """BGR frame at `seconds` into the clip (or the last one if the clip is shorter)."""
    with av.open(str(clip)) as box:
        stream = box.streams.video[0]
        last = None
        for frame in box.decode(stream):
            last = frame
            if frame.time is not None and frame.time >= seconds:
                break
        return last.to_ndarray(format="bgr24") if last is not None else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config)
    pre = {c.name: c.pre_seconds for c in cfg.cameras}
    client = OllamaClient(cfg.ai)
    roots = [cfg.clips_dir] + ([Path(cfg.archive["dir"])] if cfg.archive else [])
    todo = []
    for root in roots:
        for js in root.rglob("*.json"):
            try:
                data = json.loads(js.read_text())
            except (OSError, ValueError):
                continue
            ai = data.get("ai") or {}
            if ai.get("status") == "ok" and ai.get("prompt_version") != PROMPT_VERSION:
                todo.append(js)
    todo.sort(reverse=True)                      # newest first: those are searched most
    if args.limit:
        todo = todo[:args.limit]
    log.info("Describing %d recordings again (%s)", len(todo), PROMPT_VERSION)
    done = 0
    for js in todo:
        try:
            data = json.loads(js.read_text())
            clip = js.with_suffix(".mp4")
            frame = frame_at(clip, pre.get(data.get("camera"), 10.0))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            if data.get("box"):
                x1, y1, x2, y2 = data["box"]
                box = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
                frame = crop_around(frame, box, (h, w), margin=AI_CROP_MARGIN, min_side=int(AI_CROP_MIN * h))
            jpeg, gray = prepare_image(frame, [], int(cfg.ai["image_max_edge"]),
                                       int(cfg.ai["image_max_bytes"]), int(cfg.ai["image_max_pixels"]))
            t0 = time.perf_counter()
            result = client.describe(jpeg, gray, float(cfg.ai["request_timeout"]))
            old = data["ai"]
            data["ai"] = {"status": "ok", "duration_ms": int((time.perf_counter() - t0) * 1000), **result,
                          "model": cfg.ai["model"], "prompt_version": PROMPT_VERSION,
                          "captured": old.get("captured"),
                          "processed": datetime.now().astimezone().isoformat(timespec="seconds")}
            if data.get("false_alarm") in (None, "ai"):   # a verdict of your own ('nothing') stays
                data["false_alarm"] = "ai" if false_alarm(data.get("labels") or [data.get("label")], data["ai"]) else None
            tmp = js.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            tmp.replace(js)
            done += 1
            if done % 10 == 0:
                log.info("%d/%d done", done, len(todo))
        except FileNotFoundError:
            continue                             # archived or cleaned up in the meantime
        except (AiError, OSError, ValueError, av.FFmpegError) as exc:
            log.warning("%s skipped: %s", js.name, exc)
    log.info("Done: %d of %d described again", done, len(todo))


if __name__ == "__main__":
    main()
