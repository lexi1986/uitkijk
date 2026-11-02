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
    """30 s H.264 + AAC, keyframe elke 2 s (zoals een IP-camera)."""
    out = tmp_path_factory.mktemp("media") / "sample.mp4"
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
        "-t", "30", "-c:v", "libx264", "-g", "30", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(out)], check=True)
    return out


@pytest.fixture(scope="session")
def g711_video(tmp_path_factory) -> Path:
    """Zoals een Foscam-deurbel: H.264 + G.711 (A-law) 8 kHz mono."""
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
            t = float(p.pts * p.time_base)       # gesimuleerde aankomsttijd
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
            if writer is None and t >= 15:          # event op t=15
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
    # beweging links (gemaskeerd) telt niet
    left = bg.copy(); left[50:100, 20:80] = 255
    assert not md.is_motion(md.detect(left))
    # beweging rechts telt wel
    right = bg.copy(); right[50:100, 220:280] = 255
    assert md.is_motion(md.detect(right))
    # lichtomslag over het hele beeld telt niet
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
    assert removed == 2                          # a (te oud) en b (te groot, oudste)
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
      labels: {dier: true}
""")
    cfg = load_config(cfg_file)
    cam = cfg.cameras[0]
    assert cam.post_seconds == 20 and cam.pre_seconds == 10
    assert set(cam.enabled_labels()) == {"persoon", "voertuig", "dier"}
    assert cfg.retention_days == 7 and cfg.retention_max_gb == 25


def test_parked_car_stops_triggering():
    from types import SimpleNamespace
    from camsvc.camera import Camera
    from camsvc.detector import Detection

    cam = SimpleNamespace(cfg=SimpleNamespace(objects={"stationary_seconds": 30}), _tracks=[])
    cam._part_of_parked = lambda n, d, lim: Camera._part_of_parked(cam, n, d, lim)
    drop = lambda now, dets: Camera._drop_stationary(cam, now, dets, ["persoon", "voertuig"])
    car = Detection("voertuig", 0.7, (100, 100, 300, 250))
    jitter = Detection("voertuig", 0.7, (103, 98, 302, 252))
    person = Detection("persoon", 0.8, (400, 100, 450, 250))

    assert drop(0, [car]) == [car]               # net verschenen: telt mee
    assert drop(10, [jitter]) == [jitter]        # nog geen 30 s
    assert drop(40, [car, person]) == [person]   # auto staat stil, persoon telt altijd
    moved = Detection("voertuig", 0.7, (250, 100, 450, 250))
    assert drop(45, [moved]) == [moved]          # rijdt weg: telt weer mee
    # 's nachts lange tijd niet herkend (zekerheid onder de drempel): niet vergeten
    for t in range(50, 3000, 2):
        drop(t, [])
    assert drop(3000, [car]) == []               # nog steeds stilstaand, geen event
    drop(3000 + 3700, [])                        # > 1 uur niet gezien: vergeten
    assert drop(3000 + 3701, [car]) == [car]     # en daarna weer 'nieuw'


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
                    reason="gezichtsmodellen niet aanwezig (CAMSVC_FACE_MODELS)")
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
    before = SimpleNamespace(name="carport", _tracks=[Track("voertuig", (1, 2, 3, 4), 100.0, 390.0)],
                             _tracks_file=f, _tracks_saved=0.0)
    Camera._save_tracks(before, 400.0)                  # staat al 300 s stil
    after = SimpleNamespace(name="carport", _tracks=[], _tracks_file=f, _lock=threading.Lock())
    Camera._load_tracks(after)
    t = after._tracks[0]
    assert t.label == "voertuig" and t.box == (1, 2, 3, 4)
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
    assert normalize_nl("123456") is None                  # zou 2 correcties nodig hebben
    assert normalize_nl("6KB31R") == "GKB-31-R"            # 6 op een letterplek -> G
    assert normalize_nl("ZBO7LP") == "ZB-07-LP"            # O op een cijferplek -> 0
    assert normalize_nl("5AU5341") is None                 # 7 tekens: geen NL-kenteken
    assert normalize_nl("") is None
    assert valid_plate("GKB-31-R") and not valid_plate("GKB31R")


def test_event_search_text_and_matching(tmp_path):
    from camsvc.events import EventStore
    data = {"camera": "carport", "labels": ["voertuig", "persoon"],
            "faces": [{"name": "Sam"}], "plates": [{"plate": "GKB-31-R"}],
            "ai": {"status": "ok", "beschrijving": "Een bezorger met een pakket.", "objecten": ["bestelbus"],
                   "kenmerken": ["witte bestelbus", "oranje hesje"]}}
    js = tmp_path / "e.json"
    js.write_text(json.dumps(data))
    store = EventStore(tmp_path, "")
    names = {"GKB-31-R": "Onze auto"}
    for q in ("pakket", "sam", "gkb31r", "GKB-31-R", "onze", "witte bestelbus", "oranje carport"):
        assert store._matches(js, q.lower().split(), names), q
    assert not store._matches(js, ["kat"], names)
    assert not store._matches(js, ["pakket", "kat"], names)        # alle woorden moeten passen


def test_motion_region_zooms_on_small_motion():
    from camsvc.motion import motion_region
    # kleine beweging rechtsonder in een 320x180-bewegingsbeeld, sub-stream 640x360
    r = motion_region((280, 150, 10, 20, 320, 180), (360, 640))
    x0, y0, x1, y1 = r
    assert x1 - x0 == y1 - y0 >= 126 and x1 <= 640 and y1 <= 360    # vierkant, binnen beeld
    assert x0 <= 570 <= x1 and y0 <= 330 <= y1                       # bevat de beweging
    assert motion_region((0, 0, 300, 170, 320, 180), (360, 640)) is None   # groot: heel beeld


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
    for name in ("101010_persoon", "111111_voertuig", "121212_persoon.face1"):
        for ext in (".mp4",) if "face" not in name else (".jpg",):
            f = d / f"{name}{ext}"
            f.write_bytes(b"x")
            os.utime(f, (old, old))
    removed, _ = cleanup(tmp_path, max_days=7, max_gb=10, days_by_label={"persoon": 30})
    assert removed == 1
    assert sorted(p.name for p in d.iterdir()) == ["101010_persoon.mp4", "121212_persoon.face1.jpg"]


def test_verify_borderline_detections():
    import threading
    from types import SimpleNamespace
    import numpy as np
    from camsvc.camera import Camera, Verdict, zoom_score
    from camsvc.detector import Detection

    crops = []
    det = SimpleNamespace(detect=lambda img, labels, s: crops.append(img.shape) or [Detection("persoon", 0.9, (0, 0, 9, 9))])
    main = np.zeros((2160, 3840, 3), np.uint8)
    assert zoom_score(main, (300, 100, 320, 160), (360, 640), det, "persoon") == 0.9
    assert crops[-1][:2] == (576, 576)               # 60 px hoog x 6 x 1.6

    now = 1000.0
    cam = SimpleNamespace(cfg=SimpleNamespace(objects={"verify_below": 0.75}), _lock=threading.Lock(),
                          _verdicts=[Verdict("persoon", (0, 0, 20, 100), now + 60, False),
                                     Verdict("persoon", (300, 50, 340, 200), now + 60, True)],
                          _verify_busy=True)
    sure = Detection("persoon", 0.9, (500, 50, 540, 200))
    pipe = Detection("persoon", 0.62, (1, 0, 21, 100))        # eerder afgewezen plek
    walker = Detection("persoon", 0.6, (310, 55, 350, 205))   # bevestigd, iets verplaatst
    new = Detection("persoon", 0.6, (100, 100, 140, 250))     # wacht op controle
    car = Detection("voertuig", 0.6, (100, 100, 200, 200))
    kept = Camera._verify(cam, now, [sure, pipe, walker, new, car], (360, 640))
    assert kept == [sure, walker, car]
    cam.cfg.objects["verify_below"] = 0
    assert Camera._verify(cam, now, [pipe], (360, 640)) == [pipe]


def test_sunlit_side_of_parked_car_is_not_new():
    from types import SimpleNamespace
    from camsvc.camera import Camera, Track
    from camsvc.detector import Detection

    now = 10_000.0
    parked = Track("voertuig", (152, 49, 416, 288), now - 3600, now - 90)
    cam = SimpleNamespace(cfg=SimpleNamespace(objects={"stationary_seconds": 30}), _tracks=[parked])
    cam._part_of_parked = lambda n, d, lim: Camera._part_of_parked(cam, n, d, lim)
    side = Detection("voertuig", 0.91, (385, 73, 414, 207))          # zonnige zijkant (echte box)
    shadow = Detection("voertuig", 0.8, (150, 45, 420, 300))         # auto + schaduw
    other = Detection("voertuig", 0.9, (420, 20, 600, 120))          # andere auto op straat
    moving = Camera._drop_stationary(cam, now, [side, shadow, other], ["voertuig"])
    assert moving == [other] and parked.last_seen == now
    # auto weg en pas na lange tijd terug: dan telt het gewoon als nieuw
    parked.last_seen = now - 2000
    assert Camera._drop_stationary(cam, now, [side], ["voertuig"]) == [side]


def test_parked_car_refreshed_while_quiet():
    """Een uur geen beweging: de geparkeerde auto wordt in rust nageteld, zodat een
    langsrijdende auto die door de ruiten te zien is daarna niet als nieuw telt."""
    import threading
    from types import SimpleNamespace
    from camsvc.camera import Camera, Track
    from camsvc.detector import Detection

    now = 10_000.0
    parked = Track("voertuig", (147, 48, 416, 292), now - 16000, now - 3400)
    gone = Track("voertuig", (420, 20, 600, 120), now - 16000, now - 3400)
    detector = SimpleNamespace(detect=lambda bgr, labels, score: [
        Detection("voertuig", 0.9, (150, 50, 415, 290))])
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
    window = Detection("voertuig", 0.9, (349, 51, 414, 167))        # echte box van 15:57
    assert Camera._drop_stationary(cam, now + 20, [window], ["voertuig"]) == []


def test_zones_validate_and_hit():
    from camsvc.motion import validate_zones, zones_at

    zones = validate_zones([{"name": " Oprit ", "points": [[0, 50], [50, 50], [50, 100], [0, 100]]},
                            {"name": "Voordeur", "points": [[50, 0], [100, 0], [100, 100]]}])
    assert zones[0]["name"] == "Oprit"
    assert zones_at(zones, 640, 360, 100, 300) == ["Oprit"]          # linksonder
    assert zones_at(zones, 640, 360, 600, 50) == ["Voordeur"]
    assert zones_at(zones, 640, 360, 100, 50) == []
    for bad in ([{"name": "", "points": [[0, 0], [1, 1], [2, 0]]}],
                [{"name": "a", "points": [[0, 0], [1, 1], [2, 0]]}, {"name": "A", "points": [[0, 0], [1, 1], [2, 0]]}],
                [{"name": "x", "points": [[0, 0], [1, 1]]}], "geen lijst"):
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
    cam = SimpleNamespace(name="carport", plates=plates, _lock=threading.Lock(), _plate_watch=[],
                          mqtt=SimpleNamespace(publish_alert=lambda c, t, d: sent.append(d)),
                          store=SimpleNamespace(public_url="http://x"))
    ev = lambda h: SimpleNamespace(start_wall=datetime(2026, 11, 2, h, 0).astimezone(), vehicle_box=(10, 10, 200, 150))
    rows = [{"plate": "GKS-31-R", "name": None, "crop": "a.jpg"},     # eigen auto, één teken mis
            {"plate": "XX-123-B", "name": None, "crop": "b.jpg"}]
    Camera._check_unknown_plates(cam, ev(23), rows)
    assert [d["plate"] for d in sent] == ["XX-123-B"] and sent[0]["reden"] == "nacht"
    Camera._check_unknown_plates(cam, ev(14), rows)
    assert len(sent) == 1 and [w["plate"] for w in cam._plate_watch] == ["XX-123-B"]
    cam.latest_bgr = np.zeros((360, 640, 3), np.uint8)
    cam.cfg = SimpleNamespace(objects={"min_score": 0.5})
    cam.detector = SimpleNamespace(detect=lambda *a: [Detection("voertuig", 0.9, (12, 12, 198, 150))])
    Camera._check_parked(cam, cam._plate_watch)
    assert sent[-1]["reden"] == "geparkeerd" and sent[-1]["plate"] == "XX-123-B"


def test_day_summary(tmp_path):
    from datetime import date
    from camsvc.summary import summarize_day

    day = tmp_path / "deurbel" / "2026-11-02"
    day.mkdir(parents=True)
    rows = [
        {"camera": "deurbel", "label": "aanbellen", "labels": ["aanbellen", "persoon"], "start": "2026-11-02T10:05:00+01:00",
         "faces": [{"name": None}], "ai": {"status": "ok", "kenmerken": ["bruin pakket"], "objecten": ["persoon"]}},
        {"camera": "deurbel", "label": "persoon", "labels": ["persoon"], "start": "2026-11-02T15:42:00+01:00",
         "faces": [{"name": "Sam"}]},
        {"camera": "deurbel", "label": "voertuig", "labels": ["voertuig"], "start": "2026-11-02T16:00:00+01:00",
         "plates": [{"plate": "GKS-31-R"}, {"plate": "XX-123-B"}]},
    ]
    for i, r in enumerate(rows):
        (day / f"{i}.json").write_text(json.dumps(r))
    s = summarize_day(tmp_path, date(2026, 11, 2), {"GKB-31-R": "Family car"})
    assert s["events"] == 3 and s["aanbellen"] == 1 and s["pakketten"] == 1 and s["onbekende_personen"] == 1
    assert s["personen"] == {"Sam": {"aantal": 1, "eerst": "15:42", "laatst": "15:42"}}
    assert s["tekst"] == ("3 opnames. 1× aangebeld. 1 pakketje. 2× een persoon: Sam (1×, eerst 15:42), "
                          "1× onbekend gezicht. Auto's: Family car, onbekend XX-123-B.")
    assert summarize_day(tmp_path, date(2026, 11, 3))["tekst"] == "Niets opgenomen vandaag."


def test_likely_resident_after_arrival():
    import time as _t
    from types import SimpleNamespace
    from camsvc.camera import Camera

    now = _t.monotonic()
    cam = SimpleNamespace(arrivals={"Alex": now - 60, "Sam": now - 4000}, bodies=None, name="hek")
    cam._recent_arrivals = lambda n: Camera._recent_arrivals(cam, n)
    ev = SimpleNamespace(labels={"persoon"}, start_mono=now, body_emb=None, max_persons=1)
    assert Camera._likely_residents(cam, ev, [{"name": None}]) == ["Alex"]
    assert Camera._likely_residents(cam, ev, [{"name": "Sam"}]) is None          # gezicht herkend
    assert Camera._likely_residents(cam, SimpleNamespace(labels={"voertuig"}, start_mono=now, body_emb=None), []) is None
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
    m._lock, m._file, m._seen = threading.Lock(), tmp_path / "kleding.json", {}
    now = _t.monotonic()
    cam = SimpleNamespace(arrivals={}, bodies=m, name="hek")
    cam._recent_arrivals = lambda n: Camera._recent_arrivals(cam, n)
    # gezicht herkend bij de deurbel: kleding onthouden
    seen = SimpleNamespace(labels={"persoon"}, start_mono=now, body_emb=v(1, 0.1), max_persons=1)
    assert Camera._likely_residents(cam, seen, [{"name": "Sam"}]) is None and "Sam" in m._seen
    # later van achteren bij het hek, zelfde kleding
    back = SimpleNamespace(labels={"persoon"}, start_mono=now, body_emb=v(1, 0.3), max_persons=1)
    assert Camera._likely_residents(cam, back, [{"name": None}]) == ["Sam"]
    other = SimpleNamespace(labels={"persoon"}, start_mono=now, body_emb=v(0, 0, 1), max_persons=1)
    assert Camera._likely_residents(cam, other, []) is None
    # twee personen in beeld: niet onthouden (welke kleding hoort bij wie?)
    m._seen.clear()
    two = SimpleNamespace(labels={"persoon"}, start_mono=now, body_emb=v(1), max_persons=2)
    Camera._likely_residents(cam, two, [{"name": "Sam"}])
    assert m._seen == {}


def test_false_alarms_left_out_and_compared(tmp_path):
    from datetime import date
    from camsvc.summary import compare_ai, summarize_day

    day = tmp_path / "hek" / "2026-11-02"
    day.mkdir(parents=True)
    seen = {"status": "ok", "beeldkwaliteit": "goed", "persoon_zichtbaar": True,
            "voertuig_zichtbaar": False, "dier_zichtbaar": False, "beschrijving": "Iemand bij het hek."}
    plant = {**seen, "persoon_zichtbaar": False, "beschrijving": "Een plant in een bak."}
    rows = [
        {"camera": "hek", "label": "persoon", "labels": ["persoon"], "start": "2026-11-02T09:00:00+01:00",
         "clip": "hek/2026-11-02/a.mp4", "ai": seen, "extern": {"melden": True, "titel": "Bezoeker"}},
        {"camera": "hek", "label": "persoon", "labels": ["persoon"], "start": "2026-11-02T10:00:00+01:00",
         "clip": "hek/2026-11-02/b.mp4", "ai": plant, "vals": "ai", "extern": {"melden": False}},
        {"camera": "hek", "label": "persoon", "labels": ["persoon"], "start": "2026-11-02T11:00:00+01:00",
         "clip": "hek/2026-11-02/c.mp4", "ai": plant, "vals": "ai", "extern": {"melden": True, "titel": "Iemand"}},
    ]
    for i, r in enumerate(rows):
        (day / f"{i}.json").write_text(json.dumps(r))
    s = summarize_day(tmp_path, date(2026, 11, 2))
    assert s["events"] == 1 and s["vals"] == 2 and s["persoon_events"] == 1
    assert s["tekst"] == "1 opnames. 1× een persoon. 2× niets aan de hand weggelaten."
    c = compare_ai(tmp_path, date(2026, 11, 2))
    assert (c["events"], c["eens"], c["alleen_extern"], c["alleen_lokaal"]) == (3, 2, 1, 0)
    assert [r["clip"] for r in c["oneens"]] == ["hek/2026-11-02/c.mp4"]


def test_false_spot_blocks_same_object(tmp_path):
    import threading
    from types import SimpleNamespace
    from camsvc.camera import Camera
    from camsvc.detector import Detection

    cam = SimpleNamespace(object_active={"persoon": False, "voertuig": False, "dier": False},
                          _false_file=tmp_path / "vals_hek.json", false_spots=[], name="hek",
                          _lock=threading.RLock())
    cam._save_false_spots = lambda: Camera._save_false_spots(cam)
    cam._load_false_spots = lambda: Camera._load_false_spots(cam)
    assert Camera.add_false_spot(cam, "persoon", [0.10, 0.20, 0.20, 0.60], "ev1")
    assert not Camera.add_false_spot(cam, "aanbellen", [0, 0, 1, 1], "ev2")
    shape = (480, 640)
    pipe = Detection("persoon", 0.62, (66, 98, 128, 290))            # zelfde regenpijp
    walker = Detection("persoon", 0.9, (300, 100, 380, 400))          # iemand anders in beeld
    car = Detection("voertuig", 0.9, (66, 98, 128, 290))              # ander soort object
    assert Camera._at_false_spot(cam, pipe, shape)
    assert not Camera._at_false_spot(cam, walker, shape) and not Camera._at_false_spot(cam, car, shape)
    assert Camera._load_false_spots(cam)[0]["event"] == "ev1"          # bewaard over herstarts
    assert Camera.remove_false_spot(cam, "ev1") == 1 and cam.false_spots == []


def test_mark_false_updates_event(tmp_path):
    from types import SimpleNamespace
    from camsvc.events import EventStore
    from camsvc.service import Service

    day = tmp_path / "hek" / "2026-11-02"
    day.mkdir(parents=True)
    (day / "100000_persoon.json").write_text(json.dumps({
        "id": "ev1", "camera": "hek", "label": "persoon", "labels": ["persoon"], "score": 0.6,
        "start": "2026-11-02T10:00:00+01:00", "end": "", "duration": 30, "clip": "hek/2026-11-02/100000_persoon.mp4",
        "snapshot": None, "box": [0.1, 0.2, 0.2, 0.6]}))
    learned = []
    cam = SimpleNamespace(object_active={"persoon": False},
                          add_false_spot=lambda *a: learned.append(a) or True,
                          remove_false_spot=lambda ev: learned.clear())
    svc = SimpleNamespace(store=EventStore(tmp_path, ""), cameras={"hek": cam})
    assert Service.mark_false(svc, "hek/2026-11-02/100000_persoon.mp4", True) == {
        "vals": "gebruiker", "plek_onthouden": True}
    assert learned == [("persoon", [0.1, 0.2, 0.2, 0.6], "ev1")]
    assert json.loads((day / "100000_persoon.json").read_text())["vals"] == "gebruiker"
    assert Service.mark_false(svc, "hek/2026-11-02/100000_persoon.mp4", False)["vals"] is None
    assert learned == []
    with pytest.raises(KeyError):
        Service.mark_false(svc, "../../etc/passwd", True)


def test_health_marker_and_snapshot(tmp_path):
    from camsvc.health import Health

    h = Health(tmp_path, tmp_path)
    assert not h.previous_crash
    snap = h.snapshot("ready", 2, 3)
    assert snap["cameras_online"] == 2 and 0 <= snap["schijf"] <= 100 and snap["geheugen_mb"] >= 0  # 0 buiten Linux
    assert Health(tmp_path, tmp_path).previous_crash          # niet netjes gestopt
    Health(tmp_path, tmp_path).clean_stop()
    assert not Health(tmp_path, tmp_path).previous_crash


def test_interrupted_clip_is_playable_and_rescued(sample_video, tmp_path):
    """Gefragmenteerde MP4: de bytes op het moment van een crash zijn tot daar afspeelbaar,
    en de service maakt er bij het opstarten een gewoon event van."""
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
        shutil.copy(out, crashed)                 # zo staat het bestand er bij een crash
        writer.close()
    assert clip_duration(crashed) >= 5
    (parts / "abc123.meta.json").write_text(json.dumps(
        {"id": "abc123", "camera": "hek", "label": "persoon", "start": "2026-11-02T10:00:00+01:00"}))
    (parts / "kapot.mp4.part").write_bytes(b"geen mp4")
    svc = SimpleNamespace(cfg=SimpleNamespace(clips_dir=tmp_path), store=EventStore(tmp_path, ""))
    Service._rescue_parts(svc)
    saved = json.loads((tmp_path / "hek" / "2026-11-02" / "100000_persoon.json").read_text())
    assert saved["id"] == "abc123" and saved["duration"] >= 5
    assert list(parts.iterdir()) == []


def test_important_filter_hides_false_alarms_and_own_car(tmp_path):
    from camsvc.events import EventStore

    day = tmp_path / "carport" / "2026-11-02"
    day.mkdir(parents=True)
    base = {"camera": "carport", "score": 0.9, "end": "", "duration": 30, "snapshot": None}
    rows = {"100000_voertuig": {"label": "voertuig", "labels": ["voertuig"], "plates": [{"plate": "GKB-31-R"}]},
            "110000_voertuig": {"label": "voertuig", "labels": ["voertuig"], "plates": [{"plate": "XX-123-B"}]},
            "120000_persoon": {"label": "persoon", "labels": ["persoon"], "vals": "ai"},
            "130000_persoon": {"label": "persoon", "labels": ["persoon", "voertuig"], "plates": [{"plate": "GKB-31-R"}]}}
    for i, (stem, r) in enumerate(rows.items()):
        (day / f"{stem}.json").write_text(json.dumps({**base, **r, "id": str(i), "start": f"2026-11-02T{stem[:2]}:00:00+01:00",
                                                        "clip": f"carport/2026-11-02/{stem}.mp4"}))
    store = EventStore(tmp_path, "")
    items, total = store.browse(names={"GKB-31-R": "Onze auto"}, important=True)
    assert total == 2 and [e["id"] for e in items] == ["3", "1"]
    assert store.browse(names={"GKB-31-R": "Onze auto"})[1] == 4
