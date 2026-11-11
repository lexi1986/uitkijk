import json
import os
import subprocess
import time
from pathlib import Path

import av
import numpy as np
import pytest

from camsvc.config import load_config
from camsvc.motion import MotionDetector, build_mask, point_masked
from camsvc.recorder import BufferedPacket, ClipWriter, PacketRing
from camsvc.retention import cleanup


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    """30 s H.264 + AAC, keyframe every 2 s (like an IP camera)."""
    out = tmp_path_factory.mktemp("media") / "sample.mp4"
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
        "-t", "30", "-c:v", "libx264", "-g", "30", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(out)], check=True)
    return out


@pytest.fixture(scope="session")
def g711_video(tmp_path_factory) -> Path:
    """Like a Foscam doorbell: H.264 + G.711 (A-law) 8 kHz mono."""
    out = tmp_path_factory.mktemp("media") / "g711.mkv"
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=8000",
        "-t", "20", "-c:v", "libx264", "-g", "30", "-pix_fmt", "yuv420p",
        "-c:a", "pcm_alaw", "-ac", "1", "-shortest", str(out)], check=True)
    return out


def probe_duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def probe_streams(path: Path) -> list[str]:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return r.stdout.split()


def test_ring_keeps_pre_seconds_from_keyframe(sample_video):
    ring = PacketRing(pre_seconds=10)
    with av.open(str(sample_video)) as c:
        v = c.streams.video[0]
        for p in c.demux():
            if p.size == 0 or p.pts is None:
                continue
            t = float(p.pts * p.time_base)       # simulated arrival time
            ring.add(p.stream.index, bytes(p), p.pts, p.dts,
                     p.stream.index == v.index and p.is_keyframe, t)
    assert ring.packets[0].is_keyframe
    assert 10.0 <= ring.seconds <= 12.5


def test_clip_writer_stream_copy(sample_video, tmp_path):
    ring = PacketRing(pre_seconds=10)
    out = tmp_path / "clip.mp4"
    with av.open(str(sample_video)) as c:
        v = c.streams.video[0]
        streams = {s.index: s for s in c.streams}
        writer = None
        for p in c.demux():
            if p.size == 0 or p.pts is None:
                continue
            t = float(p.pts * p.time_base)
            bp = ring.add(p.stream.index, bytes(p), p.pts, p.dts,
                          p.stream.index == v.index and p.is_keyframe, t)
            if writer is None and t >= 15:          # event at t=15
                writer = ClipWriter(out, streams, v.index)
                for old in ring.snapshot():
                    writer.write(old)
            elif writer is not None and bp is not None and t < 25:   # 10 s post
                writer.write(bp)
        writer.close()
    dur = probe_duration(out)
    assert 19.0 <= dur <= 22.5, dur
    assert probe_streams(out) == ["h264", "aac"]


def test_clip_writer_transcodes_g711(g711_video, tmp_path):
    out = tmp_path / "clip.mp4"
    with av.open(str(g711_video)) as c:
        v = c.streams.video[0]
        streams = {s.index: s for s in c.streams}
        writer = ClipWriter(out, streams, v.index)
        for p in c.demux():
            if p.size == 0 or p.pts is None:
                continue
            writer.write(BufferedPacket(0, 0.0, p.stream.index, bytes(p), p.pts, p.dts,
                                        p.stream.index == v.index and p.is_keyframe))
        writer.close()
    assert probe_streams(out) == ["h264", "aac"]
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                        "stream=duration", "-of", "csv=p=0", str(out)],
                       capture_output=True, text=True, check=True)
    assert 19.0 <= float(r.stdout.strip()) <= 20.5


def test_motion_and_mask():
    md = MotionDetector(threshold=25, min_area=0.004, polygons=[[[0, 0], [50, 0], [50, 100], [0, 100]]])
    bg = np.zeros((180, 320), np.uint8)
    assert md.detect(bg) == 0.0
    # motion on the left (masked) does not count
    left = bg.copy(); left[50:100, 20:80] = 255
    assert not md.is_motion(md.detect(left))
    # motion on the right does count
    right = bg.copy(); right[50:100, 220:280] = 255
    assert md.is_motion(md.detect(right))
    # light change over the whole frame does not count
    assert md.detect(np.full((180, 320), 200, np.uint8)) == 0.0

    mask = build_mask([[[0, 0], [50, 0], [50, 100], [0, 100]]], 100, 100)
    assert point_masked(mask, 10, 50) and not point_masked(mask, 90, 50)


def test_retention_age_and_size(tmp_path):
    d = tmp_path / "cam" / "2026-10-01"
    d.mkdir(parents=True)
    old = time.time() - 10 * 86400
    for name, size, mtime in [("a", 100, old), ("b", 600_000, time.time() - 100),
                              ("c", 600_000, time.time())]:
        for ext in (".mp4", ".json"):
            f = d / f"{name}{ext}"
            f.write_bytes(b"x" * (size if ext == ".mp4" else 10))
            os.utime(f, (mtime, mtime))
    removed, _ = cleanup(tmp_path, max_days=7, max_gb=0.7 / 1024)  # ~0.73 MB
    assert removed == 2                          # a (too old) and b (too large, oldest)
    assert sorted(p.name for p in d.iterdir()) == ["c.json", "c.mp4"]


def test_config_defaults_and_override(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("""
cameras:
  - name: deurbel
    main_url: rtsp://x/main
    sub_url: rtsp://x/sub
    post_seconds: 20
    objects:
      labels: {animal: true}
""")
    cfg = load_config(cfg_file)
    cam = cfg.cameras[0]
    assert cam.post_seconds == 20 and cam.pre_seconds == 10
    assert set(cam.enabled_labels()) == {"person", "vehicle", "animal"}
    assert cfg.retention_days == 7 and cfg.retention_max_gb == 25


def test_parked_car_stops_triggering():
    from types import SimpleNamespace
    from camsvc.camera import Camera
    from camsvc.detector import Detection

    cam = SimpleNamespace(cfg=SimpleNamespace(objects={"stationary_seconds": 30}), _tracks=[])
    cam._part_of_parked = lambda n, d, lim: Camera._part_of_parked(cam, n, d, lim)
    drop = lambda now, dets: Camera._drop_stationary(cam, now, dets, ["person", "vehicle"])
    car = Detection("vehicle", 0.7, (100, 100, 300, 250))
    jitter = Detection("vehicle", 0.7, (103, 98, 302, 252))
    person = Detection("person", 0.8, (400, 100, 450, 250))

    assert drop(0, [car]) == [car]               # just appeared: counts
    assert drop(10, [jitter]) == [jitter]        # not 30 s yet
    assert drop(40, [car, person]) == [person]   # car stands still, person always counts
    moved = Detection("vehicle", 0.7, (250, 100, 450, 250))
    assert drop(45, [moved]) == [moved]          # drives away: counts again
    # not detected for a long time at night (confidence below threshold): do not forget
    for t in range(50, 3000, 2):
        drop(t, [])
    assert drop(3000, [car]) == []               # still stationary, no event
    drop(3000 + 3700, [])                        # not seen for > 1 hour: forgotten
    assert drop(3000 + 3701, [car]) == [car]     # and 'new' again afterwards


def test_grab_frame_decodes_newest_main_frame(sample_video):
    from camsvc.recorder import MainRecorder

    rec = MainRecorder("test", lambda: "", 10, False, Path("/nonexistent"), lambda *_: None)
    with av.open(str(sample_video)) as c:
        v = c.streams.video[0]
        rec.codec, rec._video_index, rec._extradata = v.codec_context.name, v.index, v.codec_context.extradata
        for p in c.demux(v):
            if p.size and p.pts is not None:
                rec.ring.add(v.index, bytes(p), p.pts, p.dts, p.is_keyframe, float(p.pts * p.time_base))
    rec.online = True
    frame = rec.grab_frame()
    assert frame is not None and frame.shape == (360, 640, 3)


FACE_MODELS = Path(os.environ.get("CAMSVC_FACE_MODELS", "models"))


@pytest.mark.skipif(not (FACE_MODELS / "face_recognition_sface_2021dec.onnx").exists(),
                    reason="face models not present (CAMSVC_FACE_MODELS)")
def test_face_learn_and_recognise(tmp_path):
    import cv2
    from camsvc.faces import FaceRecognizer, valid_name

    img = cv2.imread(str(FACE_MODELS / "test-face.jpg"))
    fr = FaceRecognizer(FACE_MODELS, tmp_path / "faces")
    first = fr.analyse(img)
    assert len(first) == 1 and first[0].name is None
    sid = fr.add("Lena", first[0].crop_jpg)
    again = fr.analyse(cv2.flip(img, 1))
    assert again[0].name == "Lena" and again[0].score > 0.6
    assert fr.people() == {"Lena": [sid]}
    assert fr.remove("Lena", sid) and fr.people() == {}
    assert not valid_name("../etc") and not valid_name("") and valid_name("Opa Jan")


def test_validate_mask():
    from camsvc.motion import validate_mask

    assert validate_mask([[[0, 0], [100, 0], [50.04, 12]]]) == [[[0.0, 0.0], [100.0, 0.0], [50.0, 12.0]]]
    assert validate_mask([]) == []
    for bad in (None, [[[0, 0], [1, 1]]], [[[0, 0], [1, 1], [101, 5]]], [[["a", 1], [1, 1], [2, 2]]]):
        with pytest.raises(ValueError):
            validate_mask(bad)


def test_stationary_tracks_survive_restart(tmp_path):
    import threading
    from types import SimpleNamespace
    from camsvc.camera import Camera, Track

    f = tmp_path / "stationary_carport.json"
    before = SimpleNamespace(name="carport", _tracks=[Track("vehicle", (1, 2, 3, 4), 100.0, 390.0)],
                             _tracks_file=f, _tracks_saved=0.0)
    Camera._save_tracks(before, 400.0)                  # stationary for 300 s already
    after = SimpleNamespace(name="carport", _tracks=[], _tracks_file=f, _lock=threading.Lock())
    Camera._load_tracks(after)
    t = after._tracks[0]
    assert t.label == "vehicle" and t.box == (1, 2, 3, 4)
    assert 299 <= time.monotonic() - t.first_seen <= 302
    assert 9 <= time.monotonic() - t.last_seen <= 12


def test_validate_settings():
    from camsvc.config import validate_settings

    assert validate_settings({"post_seconds": 20, "faces.enabled": True}) == {"post_seconds": 20.0, "faces.enabled": True}
    for bad in ({"post_seconds": 1}, {"post_seconds": "10"}, {"faces.enabled": 1}, {"onzin": 5}, {"motion.threshold": True}):
        with pytest.raises(ValueError):
            validate_settings(bad)


def test_normalize_nl_plates():
    from camsvc.plates import normalize_nl, valid_plate

    assert normalize_nl("GKB31R") == "GKB-31-R"            # sidecode 11
    assert normalize_nl("gkb-31-r") == "GKB-31-R"
    assert normalize_nl("12XKB7") == "12-XKB-7"            # sidecode 7
    assert normalize_nl("TN321B") == "TN-321-B"            # sidecode 9
    assert normalize_nl("123456") is None                  # would need 2 corrections
    assert normalize_nl("6KB31R") == "GKB-31-R"            # 6 in a letter position -> G
    assert normalize_nl("ZBO7LP") == "ZB-07-LP"            # O in a digit position -> 0
    assert normalize_nl("5AU5341") is None                 # 7 characters: not a Dutch plate
    assert normalize_nl("") is None
    assert valid_plate("GKB-31-R") and not valid_plate("GKB31R")


def test_event_search_text_and_matching(tmp_path):
    from camsvc.events import EventStore
    data = {"camera": "carport", "labels": ["vehicle", "person"],
            "faces": [{"name": "Sam"}], "plates": [{"plate": "GKB-31-R"}],
            "ai": {"status": "ok", "description": "A courier with a package.", "objects": ["delivery van"],
                   "attributes": ["white van", "orange vest"]}}
    js = tmp_path / "e.json"
    js.write_text(json.dumps(data))
    store = EventStore(tmp_path, "")
    names = {"GKB-31-R": "Our car"}
    for q in ("package", "sam", "gkb31r", "GKB-31-R", "our", "white van", "orange carport"):
        assert store._matches(js, q.lower().split(), names), q
    assert not store._matches(js, ["cat"], names)
    assert not store._matches(js, ["package", "cat"], names)        # all words must match


def test_motion_region_zooms_on_small_motion():
    from camsvc.motion import motion_region
    # small motion bottom right in a 320x180 motion frame, sub stream 640x360
    r = motion_region((280, 150, 10, 20, 320, 180), (360, 640))
    x0, y0, x1, y1 = r
    assert x1 - x0 == y1 - y0 >= 126 and x1 <= 640 and y1 <= 360    # square, inside the frame
    assert x0 <= 570 <= x1 and y0 <= 330 <= y1                       # contains the motion
    assert motion_region((0, 0, 300, 170, 320, 180), (360, 640)) is None   # large: whole frame


def test_motion_detector_remembers_moving_area():
    md = MotionDetector(threshold=25, min_area=0.001, polygons=[])
    a = np.zeros((180, 320), np.uint8)
    md.detect(a)
    b = a.copy(); b[100:130, 200:220] = 255
    md.detect(b)
    x, y, w, h, gw, gh = md.last_box
    assert (gw, gh) == (320, 180) and 190 <= x <= 205 and 90 <= y <= 105 and w >= 15 and h >= 25


def test_retention_per_label(tmp_path):
    d = tmp_path / "deurbel" / "2026-10-01"
    d.mkdir(parents=True)
    old = time.time() - 20 * 86400
    for name in ("101010_person", "111111_vehicle", "121212_person.face1"):
        for ext in (".mp4",) if "face" not in name else (".jpg",):
            f = d / f"{name}{ext}"
            f.write_bytes(b"x")
            os.utime(f, (old, old))
    removed, _ = cleanup(tmp_path, max_days=7, max_gb=10, days_by_label={"person": 30})
    assert removed == 1
    assert sorted(p.name for p in d.iterdir()) == ["101010_person.mp4", "121212_person.face1.jpg"]


def test_verify_borderline_detections():
    import threading
    from types import SimpleNamespace
    import numpy as np
    from camsvc.camera import Camera, Verdict, zoom_score
    from camsvc.detector import Detection

    crops = []
    det = SimpleNamespace(detect=lambda img, labels, s: crops.append(img.shape) or [Detection("person", 0.9, (0, 0, 9, 9))])
    main = np.zeros((2160, 3840, 3), np.uint8)
    assert zoom_score(main, (300, 100, 320, 160), (360, 640), det, "person") == 0.9
    assert crops[-1][:2] == (576, 576)               # 60 px hoog x 6 x 1.6

    now = 1000.0
    cam = SimpleNamespace(cfg=SimpleNamespace(objects={"verify_below": 0.75}), _lock=threading.Lock(),
                          _verdicts=[Verdict("person", (0, 0, 20, 100), now + 60, False),
                                     Verdict("person", (300, 50, 340, 200), now + 60, True)],
                          _verify_busy=True, night=False, weather={"bad": False})
    sure = Detection("person", 0.9, (500, 50, 540, 200))
    pipe = Detection("person", 0.62, (1, 0, 21, 100))        # previously rejected spot
    walker = Detection("person", 0.6, (310, 55, 350, 205))   # confirmed, moved slightly
    new = Detection("person", 0.6, (100, 100, 140, 250))     # waiting for verification
    car = Detection("vehicle", 0.6, (100, 100, 200, 200))
    kept = Camera._verify(cam, now, [sure, pipe, walker, new, car], (360, 640))
    assert kept == [sure, walker, car]
    cam.night = True                                          # at night: check sure hits too
    assert Camera._verify(cam, now, [sure, walker, car], (360, 640)) == [walker, car]
    cam.night = False
    cam.cfg.objects["verify_below"] = 0
    assert Camera._verify(cam, now, [pipe], (360, 640)) == [pipe]


def test_sunlit_side_of_parked_car_is_not_new():
    from types import SimpleNamespace
    from camsvc.camera import Camera, Track
    from camsvc.detector import Detection

    now = 10_000.0
    parked = Track("vehicle", (152, 49, 416, 288), now - 3600, now - 90)
    cam = SimpleNamespace(cfg=SimpleNamespace(objects={"stationary_seconds": 30}), _tracks=[parked])
    cam._part_of_parked = lambda n, d, lim: Camera._part_of_parked(cam, n, d, lim)
    side = Detection("vehicle", 0.91, (385, 73, 414, 207))          # sunlit side (real box)
    shadow = Detection("vehicle", 0.8, (150, 45, 420, 300))         # car + shadow
    other = Detection("vehicle", 0.9, (420, 20, 600, 120))          # another car on the street
    moving = Camera._drop_stationary(cam, now, [side, shadow, other], ["vehicle"])
    assert moving == [other] and parked.last_seen == now
    # car gone and back only after a long time: counts as new
    parked.last_seen = now - 2000
    assert Camera._drop_stationary(cam, now, [side], ["vehicle"]) == [side]


def test_parked_car_refreshed_while_quiet():
    """An hour without motion: the parked car is re-checked while idle, so a passing car seen
    through its windows afterwards does not count as new."""
    import threading
    from types import SimpleNamespace
    from camsvc.camera import Camera, Track
    from camsvc.detector import Detection

    now = 10_000.0
    parked = Track("vehicle", (147, 48, 416, 292), now - 16000, now - 3400)
    gone = Track("vehicle", (420, 20, 600, 120), now - 16000, now - 3400)
    detector = SimpleNamespace(detect=lambda bgr, labels, score: [
        Detection("vehicle", 0.9, (150, 50, 415, 290))])
    cam = SimpleNamespace(cfg=SimpleNamespace(objects={"enabled": True, "min_score": 0.5,
                                                       "stationary_seconds": 30},
                                              motion={"cooldown_seconds": 5}),
                          detector=detector, _tracks=[parked, gone], _last_motion=now - 3600,
                          _last_refresh=0.0, _lock=threading.Lock())
    cam._mark_seen = lambda n, d: Camera._mark_seen(cam, n, d)
    cam._part_of_parked = lambda n, d, lim: Camera._part_of_parked(cam, n, d, lim)
    assert Camera._refresh_due(cam, now)
    Camera._refresh_parked(cam, now, None)
    assert parked.last_seen == now and gone.last_seen == now - 3400
    assert not Camera._refresh_due(cam, now + 10)
    window = Detection("vehicle", 0.9, (349, 51, 414, 167))        # real box from 15:57
    assert Camera._drop_stationary(cam, now + 20, [window], ["vehicle"]) == []


def test_zones_validate_and_hit():
    from camsvc.motion import validate_zones, zones_at

    zones = validate_zones([{"name": " Oprit ", "points": [[0, 50], [50, 50], [50, 100], [0, 100]]},
                            {"name": "Voordeur", "points": [[50, 0], [100, 0], [100, 100]]}])
    assert zones[0]["name"] == "Oprit"
    assert zones_at(zones, 640, 360, 100, 300) == ["Oprit"]          # bottom left
    assert zones_at(zones, 640, 360, 600, 50) == ["Voordeur"]
    assert zones_at(zones, 640, 360, 100, 50) == []
    for bad in ([{"name": "", "points": [[0, 0], [1, 1], [2, 0]]}],
                [{"name": "a", "points": [[0, 0], [1, 1], [2, 0]]}, {"name": "A", "points": [[0, 0], [1, 1], [2, 0]]}],
                [{"name": "x", "points": [[0, 0], [1, 1]]}], "not a list"):
        with pytest.raises(ValueError):
            validate_zones(bad)


def test_unknown_plate_alerts():
    import threading
    from datetime import datetime
    from types import SimpleNamespace
    from camsvc.camera import Camera
    from camsvc.detector import Detection
    from camsvc.plates import PlateReader

    sent = []
    plates = SimpleNamespace(known={"GKB-31-R": "Family car"})
    plates.near_known = lambda p: PlateReader.near_known(plates, p)
    plates.known_plate = lambda p: PlateReader.known_plate(plates, p)
    cam = SimpleNamespace(name="carport", plates=plates, _lock=threading.Lock(), _plate_watch=[],
                          mqtt=SimpleNamespace(publish_alert=lambda c, t, d: sent.append(d)),
                          store=SimpleNamespace(public_url="http://x"))
    ev = lambda h: SimpleNamespace(start_wall=datetime(2026, 11, 2, h, 0).astimezone(), vehicle_box=(10, 10, 200, 150))
    rows = [{"plate": "GKS-31-R", "name": None, "crop": "a.jpg"},     # own car, one character misread
            {"plate": "XX-123-B", "name": None, "crop": "b.jpg"}]
    Camera._check_unknown_plates(cam, ev(23), rows)
    assert [d["plate"] for d in sent] == ["XX-123-B"] and sent[0]["reason"] == "night"
    Camera._check_unknown_plates(cam, ev(14), rows)
    assert len(sent) == 1 and [w["plate"] for w in cam._plate_watch] == ["XX-123-B"]
    cam.latest_bgr = np.zeros((360, 640, 3), np.uint8)
    cam.cfg = SimpleNamespace(objects={"min_score": 0.5})
    cam.detector = SimpleNamespace(detect=lambda *a: [Detection("vehicle", 0.9, (12, 12, 198, 150))])
    Camera._check_parked(cam, cam._plate_watch)
    assert sent[-1]["reason"] == "parked" and sent[-1]["plate"] == "XX-123-B"
    # regular visitor: no daytime watch, still an alert at night
    cam._plate_watch, visitor = [], [{"plate": "XX-123-B", "name": None, "crop": "b.jpg", "visitor": 4}]
    Camera._check_unknown_plates(cam, ev(14), visitor)
    assert cam._plate_watch == []
    Camera._check_unknown_plates(cam, ev(23), visitor)
    assert sent[-1]["visitor"] == 4 and sent[-1]["reason"] == "night"


def test_day_summary(tmp_path):
    from datetime import date
    from camsvc.summary import summarize_day

    day = tmp_path / "deurbel" / "2026-11-02"
    day.mkdir(parents=True)
    rows = [
        {"camera": "deurbel", "label": "doorbell", "labels": ["doorbell", "person"], "start": "2026-11-02T10:05:00+01:00",
         "faces": [{"name": None}], "ai": {"status": "ok", "attributes": ["brown package"], "objects": ["person"]}},
        {"camera": "deurbel", "label": "person", "labels": ["person"], "start": "2026-11-02T15:42:00+01:00",
         "faces": [{"name": "Sam"}]},
        {"camera": "deurbel", "label": "vehicle", "labels": ["vehicle"], "start": "2026-11-02T16:00:00+01:00",
         "plates": [{"plate": "GKS-31-R"}, {"plate": "XX-123-B"}]},
    ]
    for i, r in enumerate(rows):
        (day / f"{i}.json").write_text(json.dumps(r))
    s = summarize_day(tmp_path, date(2026, 11, 2), {"GKB-31-R": "Family car"})
    assert s["events"] == 3 and s["doorbell"] == 1 and s["packages"] == 1 and s["unknown_faces"] == 1
    assert s["people"] == {"Sam": {"count": 1, "first": "15:42", "last": "15:42"}}
    assert s["text"] == ("3 recordings. Doorbell rang 1×. 1 package. 2× a person: Sam (1×, first 15:42), "
                          "1× unknown face. Vehicles: Family car, unknown XX-123-B.")
    assert summarize_day(tmp_path, date(2026, 11, 3))["text"] == "Nothing recorded today."


def test_likely_resident_after_arrival():
    import time as _t
    from types import SimpleNamespace
    from camsvc.camera import Camera

    now = _t.monotonic()
    cam = SimpleNamespace(arrivals={"Alex": now - 60, "Sam": now - 4000}, bodies=None, name="hek",
                          faces=SimpleNamespace(backend=SimpleNamespace(learn=0.38)))
    cam._recent_arrivals = lambda n: Camera._recent_arrivals(cam, n)
    ev = SimpleNamespace(labels={"person"}, start_mono=now, body_emb=None, max_persons=1)
    assert Camera._likely_residents(cam, ev, [{"name": None}]) == ["Alex"]
    assert Camera._likely_residents(cam, ev, [{"name": "Sam", "score": 0.5}]) is None   # face recognised
    assert Camera._likely_residents(cam, SimpleNamespace(labels={"vehicle"}, start_mono=now, body_emb=None), []) == ["Alex"]
    assert Camera._likely_residents(cam, SimpleNamespace(labels={"animal"}, start_mono=now, body_emb=None), []) is None
    cam.arrivals.clear()
    assert Camera._likely_residents(cam, ev, []) is None


def test_clothing_remembered_and_matched(tmp_path):
    import time as _t
    from types import SimpleNamespace
    from camsvc.camera import Camera
    from camsvc.reid import BodyMatcher

    v = lambda *xs: np.array(xs + (0.0,) * (4 - len(xs)), np.float32) / np.linalg.norm(xs)
    m = BodyMatcher.__new__(BodyMatcher)
    import threading
    m._lock, m._file, m._seen = threading.Lock(), tmp_path / "clothing.json", {}
    now = _t.monotonic()
    cam = SimpleNamespace(arrivals={}, bodies=m, name="hek",
                          faces=SimpleNamespace(backend=SimpleNamespace(learn=0.38)))
    cam._recent_arrivals = lambda n: Camera._recent_arrivals(cam, n)
    # doubtful face: do not remember the clothing
    seen = SimpleNamespace(labels={"person"}, start_mono=now, body_emb=v(1, 0.1), max_persons=1)
    Camera._likely_residents(cam, seen, [{"name": "Sam", "score": 0.3}])
    assert m._seen == {}
    # face certainly recognised at the doorbell: remember the clothing
    assert Camera._likely_residents(cam, seen, [{"name": "Sam", "score": 0.5}]) is None and "Sam" in m._seen
    # later from behind at the gate, same clothes
    back = SimpleNamespace(labels={"person"}, start_mono=now, body_emb=v(1, 0.3), max_persons=1)
    assert Camera._likely_residents(cam, back, [{"name": None}]) == ["Sam"]
    other = SimpleNamespace(labels={"person"}, start_mono=now, body_emb=v(0, 0, 1), max_persons=1)
    assert Camera._likely_residents(cam, other, []) is None
    # two people in view: do not remember (whose clothes are whose?)
    m._seen.clear()
    two = SimpleNamespace(labels={"person"}, start_mono=now, body_emb=v(1), max_persons=2)
    Camera._likely_residents(cam, two, [{"name": "Sam", "score": 0.5}])
    assert m._seen == {}


def test_clothing_overrules_doubtful_face(tmp_path):
    import threading
    import time as _t
    from types import SimpleNamespace
    from camsvc.camera import Camera
    from camsvc.faces import FaceGroup, FaceMatch
    from camsvc.reid import BodyMatcher

    m = BodyMatcher.__new__(BodyMatcher)
    m._lock, m._file = threading.Lock(), tmp_path / "clothing.json"
    alex = np.array([1, 0, 0, 0], np.float32)
    m._seen = {"Alex": [(_t.time(), alex)]}
    cam = SimpleNamespace(bodies=m, name="carport", faces=SimpleNamespace(backend=SimpleNamespace(learn=0.38)))
    face = FaceMatch("Sam", 0.3, b"", 0.5, 0.1, None)
    ev = SimpleNamespace(body_emb=alex, max_persons=1)
    doubtful = [FaceGroup("Sam", 0.3, face, [face])]
    assert Camera._check_clothing(cam, doubtful, ev)[0].name is None
    sure = [FaceGroup("Sam", 0.5, face, [face])]
    assert Camera._check_clothing(cam, sure, ev)[0].name == "Sam"              # a certain face wins
    assert Camera._check_clothing(cam, [FaceGroup("Alex", 0.3, face, [face])], ev)[0].name == "Alex"
    two = SimpleNamespace(body_emb=alex, max_persons=2)
    assert Camera._check_clothing(cam, doubtful, two)[0].name == "Sam"         # whose clothes?


def test_archive_moves_old_events_and_store_finds_them(tmp_path):
    from camsvc.archive import Archive
    from camsvc.events import EventStore
    local, nas = tmp_path / "clips", tmp_path / "nas"
    nas.mkdir()
    day = local / "carport" / "2026-11-01"
    day.mkdir(parents=True)
    old = time.time() - 72 * 3600
    for name in ("080000_person.mp4", "080000_person.jpg", "080000_person.face1.jpg"):
        (day / name).write_bytes(b"x" * 10)
    (day / "080000_person.json").write_text(json.dumps({"camera": "carport", "label": "person"}))
    for f in day.iterdir():
        os.utime(f, (old, old))
    (day / "120000_person.json").write_text("{}")   # recent: stays local

    archive = Archive(local, nas, 48, 60, 100, lambda: None, require_mount=False)
    archive.cycle()
    assert archive.ok
    assert sorted(p.name for p in day.iterdir()) == ["120000_person.json"]
    moved = nas / "carport" / "2026-11-01" / "080000_person.mp4"
    assert moved.read_bytes() == b"x" * 10
    assert abs(moved.stat().st_mtime - old) < 2          # retention counts from the recording

    store = EventStore(local, "")
    store.archive = archive
    assert store.find("carport/2026-11-01/080000_person.mp4") == moved.resolve()
    (tmp_path / "outside.mp4").write_bytes(b"x")
    assert store.find("../outside.mp4") is None
    assert store.json_path("carport/2026-11-01/080000_person.mp4") is not None

    archive.ok = False   # NAS gone: local only
    assert store.find("carport/2026-11-01/080000_person.mp4") is None


def test_archive_needs_a_real_mount(tmp_path):
    from camsvc.archive import Archive
    (tmp_path / "nas").mkdir()
    a = Archive(tmp_path / "clips", tmp_path / "nas", 48, 60, 100, lambda: None)
    assert not a.check()


def test_store_days_counts_per_day(tmp_path):
    from camsvc.events import EventStore
    for rel in ("gate/2026-11-02/080000_person.json", "gate/2026-11-02/090000_vehicle.json",
                "carport/2026-11-03/100000_person.json"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("{}")
    store = EventStore(tmp_path, "")
    assert store.days() == [{"day": "2026-11-03", "count": 1}, {"day": "2026-11-02", "count": 2}]
    assert store.days(label="person") == [{"day": "2026-11-03", "count": 1}, {"day": "2026-11-02", "count": 1}]
    assert store.days(camera="gate") == [{"day": "2026-11-02", "count": 2}]


def test_false_alarms_left_out_and_compared(tmp_path):
    from datetime import date
    from camsvc.summary import compare_ai, summarize_day

    day = tmp_path / "hek" / "2026-11-02"
    day.mkdir(parents=True)
    seen = {"status": "ok", "image_quality": "good", "person_visible": True,
            "vehicle_visible": False, "animal_visible": False, "description": "Someone at the gate."}
    plant = {**seen, "person_visible": False, "description": "A plant in a pot."}
    rows = [
        {"camera": "hek", "label": "person", "labels": ["person"], "start": "2026-11-02T09:00:00+01:00",
         "clip": "hek/2026-11-02/a.mp4", "ai": seen, "external": {"notify": True, "title": "Visitor"}},
        {"camera": "hek", "label": "person", "labels": ["person"], "start": "2026-11-02T10:00:00+01:00",
         "clip": "hek/2026-11-02/b.mp4", "ai": plant, "false_alarm": "ai", "external": {"notify": False}},
        {"camera": "hek", "label": "person", "labels": ["person"], "start": "2026-11-02T11:00:00+01:00",
         "clip": "hek/2026-11-02/c.mp4", "ai": plant, "false_alarm": "ai", "external": {"notify": True, "title": "Iemand"}},
    ]
    for i, r in enumerate(rows):
        (day / f"{i}.json").write_text(json.dumps(r))
    s = summarize_day(tmp_path, date(2026, 11, 2))
    assert s["events"] == 1 and s["false_alarms"] == 2 and s["person_events"] == 1
    assert s["text"] == "1 recordings. 1× a person. 2× nothing to worry about left out."
    c = compare_ai(tmp_path, date(2026, 11, 2))
    assert (c["events"], c["agree"], c["only_external"], c["only_local"]) == (3, 2, 1, 0)
    assert [r["clip"] for r in c["disagree"]] == ["hek/2026-11-02/c.mp4"]


def test_false_spot_blocks_same_object(tmp_path):
    import threading
    from types import SimpleNamespace
    from camsvc.camera import Camera
    from camsvc.detector import Detection

    cam = SimpleNamespace(object_active={"person": False, "vehicle": False, "animal": False},
                          _false_file=tmp_path / "false_alarm_hek.json", false_spots=[], name="hek",
                          _lock=threading.RLock())
    cam._save_false_spots = lambda: Camera._save_false_spots(cam)
    cam._load_false_spots = lambda: Camera._load_false_spots(cam)
    assert Camera.add_false_spot(cam, "person", [0.10, 0.20, 0.20, 0.60], "ev1")
    assert not Camera.add_false_spot(cam, "doorbell", [0, 0, 1, 1], "ev2")
    shape = (480, 640)
    pipe = Detection("person", 0.62, (66, 98, 128, 290))            # same drainpipe
    walker = Detection("person", 0.9, (300, 100, 380, 400))          # someone else in view
    car = Detection("vehicle", 0.9, (66, 98, 128, 290))              # another kind of object
    assert Camera._at_false_spot(cam, pipe, shape)
    assert not Camera._at_false_spot(cam, walker, shape) and not Camera._at_false_spot(cam, car, shape)
    assert Camera._load_false_spots(cam)[0]["event"] == "ev1"          # kept across restarts
    assert Camera.remove_false_spot(cam, "ev1") == 1 and cam.false_spots == []


def test_mark_false_updates_event(tmp_path):
    from types import SimpleNamespace
    from camsvc.events import EventStore
    from camsvc.service import Service

    day = tmp_path / "hek" / "2026-11-02"
    day.mkdir(parents=True)
    (day / "100000_person.json").write_text(json.dumps({
        "id": "ev1", "camera": "hek", "label": "person", "labels": ["person"], "score": 0.6,
        "start": "2026-11-02T10:00:00+01:00", "end": "", "duration": 30, "clip": "hek/2026-11-02/100000_person.mp4",
        "snapshot": None, "box": [0.1, 0.2, 0.2, 0.6]}))
    learned = []
    cam = SimpleNamespace(object_active={"person": False},
                          add_false_spot=lambda *a: learned.append(a) or True,
                          remove_false_spot=lambda ev: learned.clear())
    svc = SimpleNamespace(store=EventStore(tmp_path, ""), cameras={"hek": cam})
    assert Service.mark_false(svc, "hek/2026-11-02/100000_person.mp4", True) == {
        "false_alarm": "user", "spot_learned": True}
    assert learned == [("person", [0.1, 0.2, 0.2, 0.6], "ev1")]
    assert json.loads((day / "100000_person.json").read_text())["false_alarm"] == "user"
    assert Service.mark_false(svc, "hek/2026-11-02/100000_person.mp4", False)["false_alarm"] is None
    assert learned == []
    with pytest.raises(KeyError):
        Service.mark_false(svc, "../../etc/passwd", True)


def test_health_marker_and_snapshot(tmp_path):
    from camsvc.health import Health

    h = Health(tmp_path, tmp_path)
    assert not h.previous_crash
    snap = h.snapshot("ready", 2, 3)
    assert snap["cameras_online"] == 2 and 0 <= snap["disk"] <= 100 and snap["memory_mb"] >= 0  # 0 buiten Linux
    assert Health(tmp_path, tmp_path).previous_crash          # did not stop cleanly
    Health(tmp_path, tmp_path).clean_stop()
    assert not Health(tmp_path, tmp_path).previous_crash


def test_interrupted_clip_is_playable_and_rescued(sample_video, tmp_path):
    """Fragmented MP4: the bytes at the moment of a crash are playable up to that point, and the
    service turns them into a normal event at startup."""
    import shutil
    from types import SimpleNamespace
    from camsvc.events import EventStore
    from camsvc.service import Service, clip_duration

    parts = tmp_path / ".parts" / "hek"
    parts.mkdir(parents=True)
    out = tmp_path / "w.mp4"
    crashed = parts / "abc123.mp4.part"
    with av.open(str(sample_video)) as c:
        v = c.streams.video[0]
        writer = ClipWriter(out, {s.index: s for s in c.streams}, v.index)
        ring = PacketRing(pre_seconds=10)
        for p in c.demux():
            if p.size == 0 or p.pts is None:
                continue
            t = float(p.pts * p.time_base)
            bp = ring.add(p.stream.index, bytes(p), p.pts, p.dts, p.stream.index == v.index and p.is_keyframe, t)
            if bp is not None:
                writer.write(bp)
            if t >= 20:
                break
        writer.out.flush() if hasattr(writer.out, "flush") else None
        shutil.copy(out, crashed)                 # this is what the file looks like after a crash
        writer.close()
    assert clip_duration(crashed) >= 5
    (parts / "abc123.meta.json").write_text(json.dumps(
        {"id": "abc123", "camera": "hek", "label": "person", "start": "2026-11-02T10:00:00+01:00"}))
    (parts / "broken.mp4.part").write_bytes(b"not an mp4")
    svc = SimpleNamespace(cfg=SimpleNamespace(clips_dir=tmp_path), store=EventStore(tmp_path, ""))
    Service._rescue_parts(svc)
    saved = json.loads((tmp_path / "hek" / "2026-11-02" / "100000_person.json").read_text())
    assert saved["id"] == "abc123" and saved["duration"] >= 5
    assert list(parts.iterdir()) == []


def test_important_filter_hides_false_alarms_and_own_car(tmp_path):
    from camsvc.events import EventStore

    day = tmp_path / "carport" / "2026-11-02"
    day.mkdir(parents=True)
    base = {"camera": "carport", "score": 0.9, "end": "", "duration": 30, "snapshot": None}
    rows = {"100000_vehicle": {"label": "vehicle", "labels": ["vehicle"], "plates": [{"plate": "GKB-31-R"}]},
            "110000_vehicle": {"label": "vehicle", "labels": ["vehicle"], "plates": [{"plate": "XX-123-B"}]},
            "120000_person": {"label": "person", "labels": ["person"], "false_alarm": "ai"},
            "130000_person": {"label": "person", "labels": ["person", "vehicle"], "plates": [{"plate": "GKB-31-R"}]}}
    for i, (stem, r) in enumerate(rows.items()):
        (day / f"{stem}.json").write_text(json.dumps({**base, **r, "id": str(i), "start": f"2026-11-02T{stem[:2]}:00:00+01:00",
                                                        "clip": f"carport/2026-11-02/{stem}.mp4"}))
    store = EventStore(tmp_path, "")
    items, total = store.browse(names={"GKB-31-R": "Our car"}, important=True)
    assert total == 2 and [e["id"] for e in items] == ["3", "1"]
    assert store.browse(names={"GKB-31-R": "Our car"})[1] == 4


def test_weather_message_sets_strict_mode():
    from types import SimpleNamespace as NS
    from camsvc.mqtt import MqttBridge
    b = MqttBridge.__new__(MqttBridge)
    b.prefix, b.discovery, b.weather = "camsvc", "homeassistant", {"bad": False}
    msg = lambda p: NS(topic="camsvc/server/weather", payload=p.encode(), retain=True)
    b._handle_message(None, None, msg('{"condition": "rainy", "wind": 10}'))
    assert b.weather["bad"] is True
    b._handle_message(None, None, msg('{"condition": "sunny", "wind": 12}'))
    assert b.weather["bad"] is False
    b._handle_message(None, None, msg('{"condition": "sunny", "wind": 42}'))   # storm without rain
    assert b.weather["bad"] is True


def test_closest_known_plate_tolerates_two_misreads():
    from camsvc.plates import closest_known

    known = {"GKB-31-R": "Family car", "XX-123-B": "Visitor"}
    assert closest_known("GKB-71-P", known) == "GKB-31-R"
    assert closest_known("GKB-37-R", known) == "GKB-31-R"
    assert closest_known("GKT-71-P", known) is None
    assert closest_known("AB-12-CD", known) is None


def test_plates_in_text():
    from camsvc.plates import plates_in_text

    assert plates_in_text("A car with plate GKB 31 R is parked on the right.") == ["GKB-31-R"]
    assert plates_in_text("A person walks by with a red bag.") == []


def test_tracker_follows_and_gives_direction():
    from types import SimpleNamespace as NS
    from camsvc.tracker import Tracker

    t = Tracker()
    a = t.update(0.0, [NS(label="person", box=(100, 100, 150, 300))])[0]
    b = t.update(1.0, [NS(label="person", box=(110, 140, 160, 340))])[0]
    assert a is b
    c = t.update(2.0, [NS(label="person", box=(120, 260, 170, 460))])[0]
    assert c is a and a.direction(640, 480) == "towards the camera"
    assert t.update(9.0, [NS(label="person", box=(120, 260, 170, 460))])[0] is not a   # gone too long


def test_archive_backs_up_state_once_a_day(tmp_path):
    import tarfile
    from camsvc.archive import BACKUP_KEEP, Archive
    state = tmp_path / "state"
    (state / "faces" / "Alex").mkdir(parents=True)
    (state / "faces" / "Alex" / "1.jpg").write_bytes(b"x")
    (state / "plates.json").write_text("{}")
    (state / "auth.json").write_text("{}")
    (state / ".cache").mkdir()
    nas = tmp_path / "nas"
    (nas / "backup").mkdir(parents=True)
    for i in range(BACKUP_KEEP + 2):
        (nas / "backup" / f"state-2000-01-{i + 1:02d}.tar.gz").write_bytes(b"")
    a = Archive(tmp_path / "clips", nas, 48, 60, 100, lambda: None, require_mount=False, state_dir=state)
    dst = a.backup_state()
    assert dst and a.backup_state() is None                     # once a day
    names = tarfile.open(dst).getnames()
    assert "plates.json" in names and "faces/Alex/1.jpg" in names
    assert "auth.json" not in names and ".cache" not in names
    assert len(list((nas / "backup").glob("state-*.tar.gz"))) == BACKUP_KEEP


def test_unknown_plate_becomes_regular_visitor(tmp_path):
    from datetime import date
    from camsvc.plates import VISITOR_DAYS, PlateReader
    r = PlateReader(tmp_path)
    d = date(2026, 11, 1)
    assert r.visit("XX-123-B", d) == 1 and r.visit("XX-123-B", d) == 1   # same day counts once
    assert r.visit("XX-123-B", date(2026, 11, 3)) == 2
    assert PlateReader(tmp_path).visit("XX-123-B", date(2026, 11, 5)) == VISITOR_DAYS   # persisted
    assert r.visit("XX-123-B", date(2027, 11, 5)) == 1                   # outside the window


def test_face_search_continues_until_everyone_in_view_is_known():
    import threading
    from types import SimpleNamespace
    from camsvc.camera import Camera
    started = []
    ev = SimpleNamespace(faces=[SimpleNamespace(name="Alex", score=0.5)], face_attempts=1, last_face_try=0.0, id="e1")
    cam = SimpleNamespace(faces=SimpleNamespace(backend=SimpleNamespace(learn=0.38)), event=ev, _face_busy=False,
                          cfg=SimpleNamespace(settings={"faces": {"attempts": 8, "interval_seconds": 1.0}}),
                          name="driveway", _face_job=lambda *a: started.append(a))
    orig = threading.Thread
    threading.Thread = lambda target, args, **k: SimpleNamespace(start=lambda: target(*args))
    try:
        Camera._maybe_faces(cam, 5.0, (360, 640), [(0, 0, 10, 10)])                    # only Alex: done
        assert not started
        Camera._maybe_faces(cam, 5.0, (360, 640), [(0, 0, 10, 10), (20, 0, 30, 10)])   # someone joins
        assert len(started) == 1 and ev.face_attempts == 2
    finally:
        threading.Thread = orig


def test_importance_rules():
    from camsvc.events import EventRecord

    def rec(label="person", start="2026-11-08T14:00:00+01:00", **kw):
        return EventRecord(id="x", camera="c", label=label, labels=[label], score=.9, start=start,
                           end=start, duration=5, clip="a.mp4", snapshot=None, **kw).importance()
    night = "2026-11-08T23:30:00+01:00"
    known = [{"name": "Sam"}]
    stranger = [{"name": None}]
    assert rec()["importance"] == "normal"
    assert rec(start=night) == {"importance": "high", "importance_reason": "unknown at night"}
    assert rec(loitering=True)["importance_reason"] == "loitering"
    assert rec(start=night, faces=known)["importance"] == "low"
    assert rec(start=night, faces=known + stranger)["importance"] == "high"
    assert rec(start=night, likely=["Sam"])["importance"] == "low"
    assert rec(start=night, false_alarm="ai")["importance"] == "low"
    assert rec("motion", night)["importance"] == "low"
    plate = [{"plate": "AB-123-C", "name": None}]
    assert rec("vehicle", night, plates=plate)["importance"] == "high"
    assert rec("vehicle", plates=plate)["importance"] == "normal"
    assert rec("vehicle", night, plates=[{"plate": "X", "visitor": 4}])["importance"] == "low"
    assert rec("vehicle", night)["importance"] == "normal"


def test_is_grayscale():
    from camsvc.faces import is_grayscale
    assert is_grayscale(np.full((8, 8, 3), 120, np.uint8))
    assert not is_grayscale(np.full((8, 8, 3), (80, 110, 170), np.uint8))


def test_confirmation_is_per_object():
    from camsvc.tracker import ObjectTrack
    box = (0, 0, 10, 20)
    o = ObjectTrack(1, "person", 0.0, 0.0, box, box)
    o.hits = [(1.0, .9)]
    assert not o.confirmed(1.0, 3, 2)              # one hit is not enough
    o.hits = [(1.0, .9), (5.0, .9)]
    assert not o.confirmed(5.0, 3, 2)              # outside the window
    o.hits = [(4.0, .9), (5.0, .5)]
    assert o.confirmed(5.0, 3, 2) and not o.confirmed(5.0, 3, 2, 0.6)
    o.hits = [(4.0, .7), (5.0, .8)]
    assert o.confirmed(5.0, 3, 2, 0.6)


def test_plausible_shape():
    from camsvc.tracker import plausible_shape as ok
    assert ok("person", (100, 100, 150, 220), 640, 360)
    assert not ok("person", (100, 100, 250, 180), 640, 360)   # much wider than tall
    assert ok("person", (0, 100, 150, 180), 640, 360)         # cut off by the edge
    assert not ok("person", (100, 100, 110, 110), 640, 360)   # tiny
    assert ok("vehicle", (100, 100, 250, 180), 640, 360)
    assert not ok("vehicle", (100, 50, 130, 200), 640, 360)   # much taller than wide


def test_unknown_person():
    from camsvc.events import unknown_person as u
    p = {"labels": ["person"]}
    assert u(p) and u({**p, "faces": [{"name": None}]})
    assert not u({**p, "faces": [{"name": "Sam"}, {"name": None}]})
    assert not u({**p, "likely": ["Sam"]})
    assert not u({**p, "false_alarm": "ai"})
    assert not u({"labels": ["vehicle"]})


def _write_events(tmp_path, events):
    day = tmp_path / "gate" / "2026-11-02"
    day.mkdir(parents=True)
    base = {"camera": "gate", "label": "person", "labels": ["person"], "score": .9, "end": "", "duration": 5,
            "snapshot": None}
    for t, extra in events.items():
        for ext in ("mp4", "jpg", "face1.jpg"):
            (day / f"{t}_person.{ext}").write_bytes(b"")
        (day / f"{t}_person.json").write_text(json.dumps(
            {**base, "id": t, "start": f"2026-11-02T{t[:2]}:00:00+01:00",
             "clip": f"gate/2026-11-02/{t}_person.mp4", **extra}))
    return day


def test_browse_person_and_plate_match_summary(tmp_path):
    from camsvc.events import EventStore
    _write_events(tmp_path, {"100000": {"faces": [{"crop": "x", "name": None}], "likely": ["Sam"]},
                             "110000": {"faces": [{"crop": "y", "name": None}]},
                             "120000": {"plates": [{"plate": "AB-123-D"}]}})   # misread AB-123-C
    store = EventStore(tmp_path, "")
    ids = lambda **kw: [e["id"] for e in store.browse(None, None, 0, 50, **kw)[0]]
    assert ids(person="Sam") == ["100000"]
    assert ids(person="unknown") == ["110000"]
    assert ids(plate="AB-123-C", names={"AB-123-C": "Our car"}) == ["120000"]


def test_delete_event_and_near_filter(tmp_path):
    from camsvc.events import EventStore
    day = _write_events(tmp_path, {"100000": {"box": [0.1, 0.1, 0.2, 0.25]}, "110000": {"box": [0.1, 0.1, 0.3, 0.8]}})
    store = EventStore(tmp_path, "")
    store.load_recent("gate")
    ids = lambda **kw: [e["id"] for e in store.browse(None, None, 0, 50, **kw)[0]]
    assert ids() == ["110000", "100000"] and ids(near=True) == ["110000"]
    gone = store.delete_event("gate/2026-11-02/110000_person.mp4")
    assert len(gone) == 4 and "gate/2026-11-02/110000_person.face1.jpg" in gone
    assert ids() == ["100000"] and [r.id for r in store.recent("gate")] == ["100000"]
    assert not [p for p in day.iterdir() if p.name.startswith("11")]
    assert store.delete_event("gate/2026-11-02/110000_person.mp4") == []
    assert store.delete_event("../../etc/passwd.mp4") == []


def test_forget_visit(tmp_path):
    import threading
    from datetime import date

    from camsvc.plates import PlateReader
    reader = PlateReader.__new__(PlateReader)
    reader._lock, reader._visits_file = threading.Lock(), tmp_path / "visitors.json"
    reader.visits = {"AB-123-C": ["2026-01-01", "2026-11-02"]}
    reader.forget_visit("AB-123-C", date(2026, 11, 2))
    assert reader.visits == {"AB-123-C": ["2026-01-01"]}
    reader.forget_visit("AB-123-C", date(2026, 1, 1))
    assert reader.visits == {} and json.loads((tmp_path / "visitors.json").read_text()) == {}


def test_loiter_check_clears_known_clothing():
    import threading
    from types import SimpleNamespace

    from camsvc.camera import Camera
    published = []

    def cam(match):
        return SimpleNamespace(
            name="c", recorder=SimpleNamespace(grab_frame=lambda: None), faces=None,
            bodies=SimpleNamespace(match=lambda emb: match), _lock=threading.Lock(),
            object_active={"person": True}, loitering=False, _loiter_busy=True, _loiter_cleared=None,
            event=None, tracker=SimpleNamespace(longest=lambda *a: 31.0),
            _publish_binary=lambda k, v: published.append((k, v)))
    ev = SimpleNamespace(id="e1", body_emb=object(), loitering=False)
    known = cam(("Sam", 0.7))
    Camera._loiter_check(known, ev)
    assert not known.loitering and known._loiter_cleared == "e1" and not known._loiter_busy and not published
    stranger = cam(None)
    stranger.event = ev
    Camera._loiter_check(stranger, ev)
    assert stranger.loitering and ev.loitering and published == [("loitering", True)]
