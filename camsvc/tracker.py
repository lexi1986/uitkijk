"""Follow moving people and vehicles per camera, each with its own number.

Simple and cheap: a detection belongs to the object with the same label that it overlaps
most (or stands close to, at a low frame rate), as long as that object was seen recently.
That tells us per object how long it was in view and which way it went, and loitering counts
per person instead of 'there was always someone'.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

MAX_GAP = 4.0          # an object may be out of sight this long and still be the same one
MIN_IOU = 0.2
MOVE_FRAC = 0.1        # displacement (fraction of the frame) below which an object 'stays put'
_ids = itertools.count(1)


@dataclass
class ObjectTrack:
    id: int
    label: str
    first_seen: float
    last_seen: float
    first_box: tuple[int, int, int, int]
    box: tuple[int, int, int, int]

    def direction(self, width: int, height: int) -> str:
        """Direction of the bottom (feet/wheels) between the first and the last position."""
        (fx, fy), (lx, ly) = _foot(self.first_box), _foot(self.box)
        dx, dy = (lx - fx) / max(width, 1), (ly - fy) / max(height, 1)
        if max(abs(dx), abs(dy)) < MOVE_FRAC:
            return "stays put"
        if abs(dy) >= abs(dx):
            return "towards the camera" if dy > 0 else "away from the camera"
        return "to the right" if dx > 0 else "to the left"


def _foot(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, box[3]


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Tracker:
    def __init__(self) -> None:
        self.active: list[ObjectTrack] = []

    def update(self, now: float, detections: list) -> list[ObjectTrack]:
        """Link detections (with .label and .box) to objects; returns per detection (in the same
        order) the object it belongs to."""
        self.active = [t for t in self.active if now - t.last_seen <= MAX_GAP]
        seen: list[ObjectTrack] = []
        of: dict[int, ObjectTrack] = {}
        for d in sorted(detections, key=lambda d: -(d.box[2] - d.box[0]) * (d.box[3] - d.box[1])):
            free = [t for t in self.active if t.label == d.label and t not in seen]
            best = max(free, key=lambda t: _iou(t.box, d.box), default=None)
            if best is None or _iou(best.box, d.box) < MIN_IOU:
                # at a low frame rate a walking person sometimes no longer overlaps: then take the
                # nearest object, if it is within one box width
                size = max(d.box[2] - d.box[0], d.box[3] - d.box[1])
                (dx, dy) = _foot(d.box)
                near = [t for t in free if abs(_foot(t.box)[0] - dx) + abs(_foot(t.box)[1] - dy) < size]
                best = min(near, key=lambda t: abs(_foot(t.box)[0] - dx) + abs(_foot(t.box)[1] - dy),
                           default=None)
            if best is None:
                best = ObjectTrack(next(_ids), d.label, now, now, d.box, d.box)
                self.active.append(best)
            best.box, best.last_seen = d.box, now
            seen.append(best)
            of[id(d)] = best
        return [of[id(d)] for d in detections]

    def longest(self, label: str, now: float) -> float:
        """How long the longest-present object of this label has been in view (0 = none)."""
        return max((now - t.first_seen for t in self.active
                    if t.label == label and now - t.last_seen <= MAX_GAP), default=0.0)
