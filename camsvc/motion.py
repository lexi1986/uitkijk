"""Bewegingsdetectie op kleine grijswaardenframes, en maskers."""

from __future__ import annotations

import cv2
import numpy as np


def build_mask(polygons: list[list[list[float]]], width: int, height: int) -> np.ndarray:
    """255 = meenemen, 0 = negeren. Punten zijn procenten (0-100) van het beeld."""
    mask = np.full((height, width), 255, dtype=np.uint8)
    for poly in polygons or []:
        pts = np.array([[x / 100.0 * width, y / 100.0 * height] for x, y in poly], dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 0)
    return mask


def point_masked(mask: np.ndarray, x: float, y: float) -> bool:
    h, w = mask.shape[:2]
    xi = min(max(int(x), 0), w - 1)
    yi = min(max(int(y), 0), h - 1)
    return mask[yi, xi] == 0


def validate_mask(polygons) -> list[list[list[float]]]:
    """Controleert een masker uit de webpagina; geeft het opgeschoonde masker terug."""
    if not isinstance(polygons, list) or len(polygons) > 20:
        raise ValueError("Een masker is een lijst van maximaal 20 vlakken")
    out = []
    for poly in polygons:
        if not isinstance(poly, list) or not 3 <= len(poly) <= 50:
            raise ValueError("Elk vlak heeft 3 tot 50 punten")
        pts = []
        for pt in poly:
            if not (isinstance(pt, list) and len(pt) == 2
                    and all(isinstance(v, (int, float)) and 0 <= v <= 100 for v in pt)):
                raise ValueError("Punten zijn [x, y] in procenten (0-100)")
            pts.append([round(float(pt[0]), 1), round(float(pt[1]), 1)])
        out.append(pts)
    return out


def draw_mask_overlay(bgr: np.ndarray, polygons: list[list[list[float]]],
                      zones: list[dict] | None = None) -> np.ndarray:
    """Tekent het masker rood en de zones violet transparant over een frame (controle via de API)."""
    out = bgr.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    for poly, color in ([(p, (0, 0, 255)) for p in polygons or []]
                        + [(z["points"], (255, 132, 156)) for z in zones or []]):
        pts = np.array([[x / 100.0 * w, y / 100.0 * h] for x, y in poly], dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(out, [pts], True, color, 2)
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
    # raster in procenten, handig om coördinaten af te lezen
    for pct in range(10, 100, 10):
        x, y = int(w * pct / 100), int(h * pct / 100)
        cv2.line(out, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.line(out, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(out, str(pct), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(out, str(pct), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return out


ZONE_NAME_MAX = 30


def validate_zones(zones) -> list[dict]:
    """Controleert zones uit de webpagina: [{"name": str, "points": [[x, y], ...]}]."""
    if not isinstance(zones, list) or len(zones) > 10:
        raise ValueError("Zones zijn een lijst van maximaal 10 vlakken")
    out, names = [], set()
    for z in zones:
        if not isinstance(z, dict):
            raise ValueError("Een zone heeft een naam en punten")
        name = str(z.get("name", "")).strip()
        if not name or len(name) > ZONE_NAME_MAX or name.lower() in names:
            raise ValueError("Elke zone heeft een eigen naam (max. 30 tekens)")
        names.add(name.lower())
        out.append({"name": name, "points": validate_mask([z.get("points")])[0]})
    return out


def zones_at(zones: list[dict], width: int, height: int, x: float, y: float) -> list[str]:
    """Namen van de zones waar punt (x, y) in pixels in valt."""
    px, py = x / width * 100.0, y / height * 100.0
    hits = []
    for z in zones:
        pts = np.array(z["points"], dtype=np.float32)
        if cv2.pointPolygonTest(pts, (px, py), False) >= 0:
            hits.append(z["name"])
    return hits


def motion_region(box: tuple[int, int, int, int, int, int], shape: tuple[int, int],
                  margin: float = 1.5, min_frac: float = 0.35,
                  max_area: float = 0.5) -> tuple[int, int, int, int] | None:
    """Vierkant gebied rond de beweging in een beeld met afmeting `shape` (hoogte, breedte),
    om het objectmodel daar in te laten zoomen. None als de beweging (met marge) toch al
    een groot deel van het beeld beslaat: dan het hele beeld."""
    bx, by, bw, bh, gw, gh = box
    H, W = shape
    sx, sy = W / gw, H / gh
    cx, cy = (bx + bw / 2) * sx, (by + bh / 2) * sy
    side = max(bw * sx * margin, bh * sy * margin, min(W, H) * min_frac)
    if side * side > max_area * W * H:
        return None
    half = side / 2
    x0, y0 = int(max(0, min(W - side, cx - half))), int(max(0, min(H - side, cy - half)))
    return x0, y0, int(min(W, x0 + side)), int(min(H, y0 + side))


class MotionDetector:
    """Frame differencing tegen een langzaam bijgewerkte achtergrond."""

    def __init__(self, threshold: int, min_area: float, polygons: list[list[list[float]]]):
        self.threshold = threshold
        self.min_area = min_area
        self.polygons = polygons
        self._bg: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._kernel = np.ones((3, 3), np.uint8)
        # gebied met de laatste echte beweging: (x, y, w, h, beeldbreedte, beeldhoogte)
        self.last_box: tuple[int, int, int, int, int, int] | None = None

    def detect(self, gray: np.ndarray) -> float:
        """Geeft het deel (0-1) van het niet-gemaskeerde beeld dat beweegt."""
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self._mask is None or self._mask.shape != gray.shape:
            self._mask = build_mask(self.polygons, gray.shape[1], gray.shape[0])
            self._bg = None
        if self._bg is None:
            self._bg = gray.astype(np.float32)
            return 0.0
        diff = cv2.absdiff(gray, cv2.convertScaleAbs(self._bg))
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_and(thresh, self._mask)
        thresh = cv2.dilate(thresh, self._kernel, iterations=2)
        # achtergrond sneller aanpassen bij lichtverandering over het hele beeld
        fraction = cv2.countNonZero(thresh) / max(cv2.countNonZero(self._mask), 1)
        alpha = 0.3 if fraction > 0.6 else 0.05
        cv2.accumulateWeighted(gray, self._bg, alpha)
        if fraction > 0.6:
            return 0.0  # vrijwel het hele beeld verandert: lichtomslag (IR aan/uit), geen beweging
        if fraction >= self.min_area:
            x, y, w, h = cv2.boundingRect(thresh)
            self.last_box = (x, y, w, h, gray.shape[1], gray.shape[0])
        return fraction

    def is_motion(self, fraction: float) -> bool:
        return fraction >= self.min_area
