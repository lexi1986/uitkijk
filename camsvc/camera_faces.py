"""Camera: faces, clothing and who it probably is (live and after saving)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import threading
import time

import av
import numpy as np

from .events import EventRecord, unknown_person
from .faces import FaceGroup, FaceMatch, FaceRecognizer, confirmed_weak, summarize
from .camera_base import (
    ARRIVAL_WINDOW, ASK_FACE_AT, ActiveEvent, BODY_ATTEMPTS, BODY_INTERVAL, BODY_MIN_FRAC, RESCAN_BODY_STEP,
    RESCAN_MAX_SECONDS, RESCAN_MIN_FRAMES, RESCAN_STEP, UNKNOWN_PERSON_DELAY, _RESCAN_SLOT, _RESCAN_WAITING,
    log,
)


class FacesMixin:
    """Methods of Camera; uses the fields Camera.__init__ sets."""

    def _publish_seen(self, start: str, clip: str, faces: list[dict], likely: list[str] | None) -> None:
        """'Last seen' per person to HA: sure by face, or likely."""
        if not self.mqtt:
            return
        sure = {f["name"] for f in faces or [] if f.get("name")}
        for name in sure | set(likely or []):
            self.mqtt.publish_seen(name, {"time": start, "camera": self.cfg.friendly_name,
                                          "clip_url": f"{self.store.public_url}/clips/{clip}",
                                          "sure": name in sure})

    def _maybe_faces(self, now: float, sub_shape: tuple[int, int],
                     boxes: list[tuple[int, int, int, int]]) -> None:
        """Start face recognition in the background (lock already held)."""
        ev = self.event
        fcfg = self.cfg.settings["faces"]
        if (self.faces is None or ev is None or not boxes or self._face_busy
                or ev.face_attempts >= int(fcfg["attempts"])
                or now - ev.last_face_try < float(fcfg["interval_seconds"])
                or len({f.name for f in ev.faces if f.name and f.score >= self.faces.backend.learn})
                >= len(boxes)):
            return                  # only stop once someone is recognised with confidence
        ev.face_attempts += 1
        ev.last_face_try = now
        self._face_busy = True
        threading.Thread(target=self._face_job, args=(ev.id, sub_shape, boxes),
                         name=f"faces-{self.name}", daemon=True).start()

    def _face_job(self, event_id: str, sub_shape: tuple[int, int],
                  boxes: list[tuple[int, int, int, int]]) -> None:
        try:
            frame = None
            if self.cfg.settings["faces"]["source"] == "main":
                frame = self.recorder.grab_frame()
            if frame is None and self.latest_bgr is not None:
                frame = self.latest_bgr.copy()
            if frame is None:
                return
            sy, sx = frame.shape[0] / sub_shape[0], frame.shape[1] / sub_shape[1]
            matches: list[FaceMatch] = []
            if not boxes:                       # no known person position (doorbell): whole frame
                matches = self.faces.analyse(frame)
            for x1, y1, x2, y2 in boxes:
                # upper part of the person, slightly wider: that is where the head is
                bw, bh = x2 - x1, y2 - y1
                cx1 = int(max(0, (x1 - 0.2 * bw) * sx))
                cx2 = int(min(frame.shape[1], (x2 + 0.2 * bw) * sx))
                cy1 = int(max(0, (y1 - 0.1 * bh) * sy))
                cy2 = int(min(frame.shape[0], (y1 + 0.6 * bh) * sy))
                if cx2 - cx1 < 40 or cy2 - cy1 < 40:
                    continue
                matches += self.faces.analyse(frame[cy1:cy2, cx1:cx2])
            if not matches and boxes:
                # nothing in the crop (head at the edge of the box, filmed from above):
                # search the whole sharp frame instead
                matches = self.faces.analyse(frame)
            if not matches:
                return
            with self._lock:
                ev = self.event if self.event and self.event.id == event_id else None
                if ev is not None:
                    # keep all attempts together (best images first); the choice is made per person
                    ev.faces = sorted(ev.faces + matches, key=lambda m: -m.quality)[:16]
                    matches = ev.faces
            groups = summarize(matches, self.faces.backend.group)
            known = next((g for g in groups if g.name), None)
            name = known.name if known else "unknown"
            log.info("[%s] face: %s%s", self.name, name,
                     f" ({known.score:.2f}, {len(known.members)} image(s))" if known else "")
            self._set_face(name)
        except Exception:  # noqa: BLE001 - recognition must never disturb the camera
            log.exception("[%s] face recognition failed", self.name)
        finally:
            self._face_busy = False

    def _maybe_body(self, now: float, sub_shape: tuple[int, int],
                    boxes: list[tuple[int, int, int, int]]) -> None:
        """(lock already held) Extract the features of the largest person from the sharp image."""
        ev = self.event
        if (self.bodies is None or ev is None or not boxes or self._body_busy
                or ev.body_attempts >= BODY_ATTEMPTS or now - ev.last_body_try < BODY_INTERVAL):
            return
        ev.body_attempts += 1
        ev.last_body_try = now
        self._body_busy = True
        box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        threading.Thread(target=self._body_job, args=(ev.id, sub_shape, box),
                         name=f"body-{self.name}", daemon=True).start()

    def _body_job(self, event_id: str, sub_shape: tuple[int, int], box: tuple[int, int, int, int]) -> None:
        try:
            frame = self.recorder.grab_frame()
            if frame is None:
                return
            sy, sx = frame.shape[0] / sub_shape[0], frame.shape[1] / sub_shape[1]
            x1, y1, x2, y2 = int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy)
            if y2 - y1 < BODY_MIN_FRAC * frame.shape[0]:
                return
            emb = self.bodies.embed(frame[y1:y2, x1:x2])
            if emb is None:
                return
            area = (x2 - x1) * (y2 - y1)
            with self._lock:
                ev = self.event if self.event and self.event.id == event_id else None
                if ev is not None and area > ev.body_area:       # the largest (closest) image counts
                    ev.body_emb, ev.body_area = emb, area
        except Exception:  # noqa: BLE001 - must never disturb the camera
            log.exception("[%s] determining clothing/build failed", self.name)
        finally:
            self._body_busy = False

    def _set_face(self, name: str) -> None:
        if name != self.face_name or name != "none":
            self.face_name = name
            if self.mqtt and self.faces is not None:
                self.mqtt.publish_face(self.name, name)

    def _rescan_faces(self, ev: ActiveEvent, mp4: Path, day_dir: Path, base: str,
                      had_faces: bool) -> None:
        """After saving, search the whole clip once more: live there are only a few attempts,
        and a face that briefly looks at the camera often falls in between."""
        recognizer = self.faces
        try:
            with _RESCAN_SLOT:
                found: list[FaceMatch] = []
                weak: list[tuple[float, FaceMatch]] = []
                last = last_body = -RESCAN_BODY_STEP
                body_area = start_area = ev.body_area
                with av.open(str(mp4)) as container:
                    # without thread_type="AUTO": with it, breaking off after RESCAN_MAX_SECONDS hangs
                    for frame in container.decode(container.streams.video[0]):
                        t = float(frame.time or 0)
                        if t > RESCAN_MAX_SECONDS:
                            break
                        if t - last < RESCAN_STEP:
                            continue
                        last = t
                        bgr = frame.to_ndarray(format="bgr24")
                        for m in recognizer.analyse(bgr, keep_weak=True):
                            if m.weak:
                                weak.append((t, m))
                            else:
                                found.append(m)
                        if t - last_body >= RESCAN_BODY_STEP:
                            last_body = t
                            body_area = self._rescan_body(ev, bgr, body_area)
            found += confirmed_weak(weak)
            body_better = body_area > start_area
            if not found and not body_better:
                return
            ev.body_area = body_area
            before = summarize(ev.faces, recognizer.backend.group)
            live = {id(m) for m in ev.faces}
            # a real face appears in several frames; a backpack or fabric pattern that looks like
            # a face in only one: such a single new hit only counts when it is certain
            groups = [g for g in summarize(ev.faces + found, recognizer.backend.group, limit=16)
                      if len(g.members) >= RESCAN_MIN_FRAMES or any(id(m) in live for m in g.members)
                      or (g.name and g.score >= recognizer.backend.learn)][:4]
            groups = self._check_clothing(groups, ev)
            old = {(g.name, g.best.quality) for g in self._check_clothing(before, ev)}
            if {(g.name, g.best.quality) for g in groups} == old and not body_better:
                return
            # faces saved live keep their file name: unknown faces and names refer to it;
            # new faces get the next number
            faces, used = [], set()
            n = len(set(ev.face_crops.values()))
            for g in groups:
                rel = next((ev.face_crops[id(m)] for m in g.members if id(m) in ev.face_crops), None)
                if rel is None or rel in used:
                    n += 1
                    rel = str((day_dir / f"{base}.face{n}.jpg").relative_to(self.clips_dir))
                used.add(rel)
                (self.clips_dir / rel).write_bytes(g.best.crop_jpg)
                faces.append({"crop": rel, "name": g.name, "score": g.score, "quality": g.best.quality})
            js = self.store.json_path(str(mp4.relative_to(self.clips_dir)))
            likely = self._likely_residents(ev, faces)
            if js:
                self.store.update(js, {"faces": faces, "likely": likely})
            if self.mqtt:
                self._publish_seen(ev.start_wall.isoformat(timespec="seconds"),
                                   str(mp4.relative_to(self.clips_dir)), faces, likely)
            log.info("[%s] clip searched again: %s", self.name,
                     ", ".join(g.name or "unknown" for g in groups))
            # learn names; only collect unknowns if there was no face at first (else they exist)
            pairs = [(g, f) for g, f in zip(groups, faces) if g.name or not had_faces]
            if pairs:
                self._learn_faces(recognizer, [g for g, _ in pairs], [f for _, f in pairs],
                                  ev.start_wall)
        except Exception:  # noqa: BLE001 - must never get in the way of saving
            log.exception("[%s] searching the clip again failed", self.name)
        finally:
            _RESCAN_WAITING.release()
            self._later_check_unknown_person(mp4)

    def _later_check_unknown_person(self, mp4: Path) -> None:
        """Alert only once the AI is done too: it can still call the event a false alarm."""
        timer = threading.Timer(UNKNOWN_PERSON_DELAY, self._check_unknown_person, args=(mp4,))
        timer.daemon = True
        timer.start()

    def _check_unknown_person(self, mp4: Path) -> None:
        try:
            data = json.loads(mp4.with_suffix(".json").read_text())
            if self.mqtt and unknown_person(data):
                log.info("[%s] unknown person after rescan: %s", self.name, mp4.name)
                self.mqtt.publish_alert(self.name, "unknown_person",
                                        {"start": data["start"], **EventRecord(**data).urls(self.store.public_url)})
        except (OSError, ValueError, TypeError):
            log.exception("[%s] unknown-person check failed", self.name)

    def _rescan_body(self, ev: ActiveEvent, bgr: np.ndarray, best_area: int) -> int:
        """Clothing from the saved clip: the largest person in this frame, if larger than what
        we had (closer = more reliable). Returns the largest area."""
        if self.bodies is None or self.detector is None:
            return best_area
        people = self.detector.detect(bgr, ["person"], float(self.cfg.objects["min_score"]))
        if not people:
            return best_area
        x1, y1, x2, y2 = max((d.box for d in people), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        area = (x2 - x1) * (y2 - y1)
        if area <= best_area or y2 - y1 < BODY_MIN_FRAC * bgr.shape[0]:
            return best_area
        emb = self.bodies.embed(bgr[max(0, y1):y2, max(0, x1):x2])
        if emb is None:
            return best_area
        ev.body_emb = emb
        return area

    def _check_clothing(self, groups: list[FaceGroup], ev: ActiveEvent) -> list[FaceGroup]:
        """Hold a doubtful name (face below the learning threshold) against today's clothing:
        if the clothing clearly matches someone else, no name. Only with one person in view,
        otherwise the clothing may belong to someone else."""
        if (self.bodies is None or self.faces is None or ev.body_emb is None
                or ev.max_persons > 1 or not any(g.name for g in groups)):
            return groups
        hit = self.bodies.match(ev.body_emb)
        if hit is None:
            return groups
        out = []
        for g in groups:
            if g.name and g.name != hit[0] and g.score < self.faces.backend.learn:
                log.info("[%s] face %s (%.2f) does not match the clothing of %s (%.2f): no name",
                         self.name, g.name, g.score, *hit)
                g = replace(g, name=None)
            out.append(g)
        return out

    def _loiter_check(self, ev: ActiveEvent | None) -> None:
        """Just before 'loitering': search the sharp image for a known face and compare the clothing
        with today's. Only if that finds nobody does loitering turn on; if the look fails, it turns
        on as well (rather one alert too many than a missed one)."""
        who = None
        try:
            frame = self.recorder.grab_frame()
            if frame is not None and self.faces is not None:
                known = next((g for g in summarize(self.faces.analyse(frame), self.faces.backend.group)
                              if g.name), None)
                if known:
                    who = known.name
                    self._set_face(who)
            if who is None and self.bodies is not None and ev is not None and ev.body_emb is not None:
                hit = self.bodies.match(ev.body_emb)
                who = hit[0] if hit else None
        except Exception:  # noqa: BLE001
            log.exception("[%s] check before loitering failed", self.name)
        try:
            with self._lock:
                if who:
                    self._loiter_cleared = ev.id if ev is not None else None
                    log.info("[%s] no loitering: probably %s", self.name, who)
                elif self.object_active["person"] and not self.loitering:
                    self.loitering = True
                    if self.event is not None:
                        self.event.loitering = True
                    log.info("[%s] person loitering (%.0f s in view)", self.name,
                             self.tracker.longest("person", time.monotonic()))
                    self._publish_binary("loitering", True)
        finally:
            self._loiter_busy = False

    def _recent_arrivals(self, now: float) -> list[str]:
        """Residents who just arrived home according to HA (within ARRIVAL_WINDOW)."""
        return sorted(n for n, t in self.arrivals.items() if 0 <= now - t <= ARRIVAL_WINDOW)

    def _likely_residents(self, ev: ActiveEvent, faces: list[dict]) -> list[str] | None:
        """Person without a recognised face: probably a resident who just arrived home (HA), or
        someone recognised by face earlier today who wears the same clothes. With a recognised
        face (and one person in view) we remember the clothing instead."""
        names = [f["name"] for f in faces if f["name"]]
        if "person" not in ev.labels:
            # only a car (resident still in the car, or not seen as a person at night):
            # whoever just arrived home per HA
            if "vehicle" in ev.labels:
                return sorted(set(self._recent_arrivals(ev.start_mono) or self._recent_arrivals(time.monotonic()))) or None
            return None
        if names:
            # only with a certain face: a doubtful name would teach the wrong clothing
            sure = {f["name"] for f in faces if f["name"] and self.faces is not None
                    and f["score"] >= self.faces.backend.learn}
            if (self.bodies is not None and ev.body_emb is not None and ev.max_persons == 1
                    and len(set(names)) == 1 and sure):
                self.bodies.remember(names[0], ev.body_emb)
            return None
        likely = set(self._recent_arrivals(ev.start_mono) or self._recent_arrivals(time.monotonic()))
        if self.bodies is not None and ev.body_emb is not None:
            hit = self.bodies.match(ev.body_emb)
            if hit:
                log.info("[%s] probably %s (clothing/build %.2f)", self.name, *hit)
                likely.add(hit[0])
        return sorted(likely) or None

    def _learn_faces(self, recognizer: FaceRecognizer, groups: list, faces: list[dict],
                     when: datetime) -> None:
        """After an event: learn confidently recognised faces, collect unknown ones."""
        fcfg = self.cfg.settings["faces"]
        for g, f in zip(groups, faces):
            try:
                if g.name and fcfg.get("auto_learn", True):
                    recognizer.learn_auto(g)
                elif not g.name and fcfg.get("collect_unknown", True):
                    sid = recognizer.unknown.add(g.best, self.name, when, f["crop"])
                    group = recognizer.unknown.group_of(sid) if sid else None
                    # ask who it is: the first time and again when someone keeps coming
                    if group and group["count"] in ASK_FACE_AT and self.mqtt:
                        self.mqtt.publish_alert(self.name, "unknown_face", {
                            "sample": sid, "count": group["count"], "suggestion": group["suggest"],
                            "names": sorted(recognizer.people())[:8]})
            except Exception:  # noqa: BLE001 - learning must never get in the way of saving
                log.exception("[%s] learning the face failed", self.name)
