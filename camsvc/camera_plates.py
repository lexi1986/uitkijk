"""Camera: reading plates, unknown vehicles and parked known cars."""

from __future__ import annotations

from datetime import datetime
import json
import threading
import time

from .events import NIGHT_HOURS
from .plates import PlateRead, vote
from .camera_base import (
    ActiveEvent, PLATE_PARKED_AFTER, STATIONARY_REFRESH, contained, iou, log,
)


class PlatesMixin:
    """Methods of Camera; uses the fields Camera.__init__ sets."""

    def _check_cars(self, now: float) -> None:
        """Known cars home or away, from their learned parking spot (lock already held)."""
        limit = float(self.cfg.objects["stationary_seconds"])
        parked = [t for t in self._tracks if t.label == "vehicle" and now - t.first_seen >= limit
                  and now - t.last_seen < STATIONARY_REFRESH + 60]
        for plate, (box, seen) in list(self._spot_pending.items()):
            spot = max(parked, key=lambda t: iou(t.box, box), default=None)
            if spot is not None and iou(spot.box, box) >= 0.3:
                self._spots[plate] = list(spot.box)
                del self._spot_pending[plate]
                try:
                    self._spots_file.write_text(json.dumps(self._spots))
                except OSError as exc:
                    log.warning("[%s] saving the parking spot failed: %s", self.name, exc)
                log.info("[%s] learned the parking spot of %s", self.name, plate)
            elif now - seen > 900:
                del self._spot_pending[plate]
        for plate, box in self._spots.items():
            home = any(iou(t.box, tuple(box)) >= 0.5 for t in parked)
            if self.car_home.get(plate) != home:
                self.car_home[plate] = home
                if self.mqtt and self.plates:
                    self.mqtt.publish_car(self.name, plate, self.plates.name_of(plate) or plate, home,
                                          datetime.now().astimezone().isoformat(timespec="seconds"))

    def _maybe_plates(self, now: float, sub_shape: tuple[int, int],
                      boxes: list[tuple[int, int, int, int]]) -> None:
        """Start a plate reading in the background (lock already held)."""
        ev = self.event
        pcfg = self.cfg.settings["plates"]
        # a known car already standing on its own spot does not need reading
        parked = [tuple(b) for p, b in self._spots.items() if self.car_home.get(p)]
        boxes = [b for b in boxes if not any(iou(b, s) >= 0.5 for s in parked)]
        if (self.plates is None or ev is None or not boxes or self._plate_busy
                or ev.plate_attempts >= int(pcfg["attempts"])
                or now - ev.last_plate_try < float(pcfg["interval_seconds"])):
            return
        # the same plate read twice: certain enough
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
            for r in reads:   # fix a misread of a known plate, so the votes add up
                r.plate = self.plates.known_plate(r.plate) or r.plate
            if not reads:
                return
            with self._lock:
                ev = self.event if self.event and self.event.id == event_id else None
                if ev is None:
                    return
                ev.plates = (ev.plates + reads)[-12:]
                best = vote(ev.plates)
            name = self.plates.name_of(best.plate)
            log.info("[%s] plate: %s%s (%.2f)", self.name, best.plate,
                     f" = {name}" if name else "", best.confidence)
            self._set_plate(best.plate, name)
        except Exception:  # noqa: BLE001 - recognition must never disturb the camera
            log.exception("[%s] plate recognition failed", self.name)
        finally:
            self._plate_busy = False

    def _check_unknown_plates(self, ev: ActiveEvent, plates: list[dict]) -> None:
        """Unknown plate: alert immediately at night, during the day only when the vehicle stays."""
        if self.plates is None or not self.mqtt:
            return
        for p in plates:
            # only named by the AI: may be a mistake, so no alert
            if p["name"] or not p.get("crop") or self.plates.near_known(p["plate"]):
                continue
            info = {"plate": p["plate"], "crop": p["crop"], "start": ev.start_wall.isoformat(timespec="seconds"),
                    "snapshot_url": f"{self.store.public_url}/clips/{p['crop']}"}
            hour = ev.start_wall.hour
            if hour >= NIGHT_HOURS[0] or hour < NIGHT_HOURS[1]:
                log.info("[%s] unknown vehicle %s at night", self.name, p["plate"])
                self.mqtt.publish_alert(self.name, "unknown_vehicle",
                                        {**info, "reason": "night", "visitor": p.get("visitor", 0)})
            elif p.get("visitor"):   # regular visitor parking during the day: no alert
                continue
            elif ev.vehicle_box is not None:
                with self._lock:
                    self._plate_watch.append({**info, "box": ev.vehicle_box,
                                              "due": time.monotonic() + PLATE_PARKED_AFTER})

    def _check_plate_watch(self, now: float) -> None:
        """(lock already held) Unknown vehicles whose quarter of an hour is up: check in the
        background whether a vehicle is still at that spot in the current frame."""
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
            cars = self.detector.detect(bgr.copy(), ["vehicle"], float(self.cfg.objects["min_score"]))
            for w in due:
                if any(iou(c.box, w["box"]) >= 0.3 or contained(c.box, w["box"]) >= 0.8
                       or contained(w["box"], c.box) >= 0.8 for c in cars) and self.mqtt:
                    log.info("[%s] unknown vehicle %s has been there for 15 minutes", self.name, w["plate"])
                    self.mqtt.publish_alert(self.name, "unknown_vehicle",
                                            {k: v for k, v in w.items() if k not in ("box", "due")}
                                            | {"reason": "parked", "minutes": int(PLATE_PARKED_AFTER // 60)})
        except Exception:  # noqa: BLE001
            log.exception("[%s] checking the parked vehicle failed", self.name)
