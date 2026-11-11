"""Faces: choice per event, grouping unknown faces and naming them (without real models)."""

import json
from datetime import datetime, timedelta

import numpy as np

from camsvc.events import EventStore
from camsvc.faces import ARCFACE, FaceMatch, UnknownFaces, summarize


def vec(*xs):
    v = np.zeros(8, np.float32)
    v[: len(xs)] = xs
    return v / np.linalg.norm(v)


A, A2, B = vec(1, 0.1), vec(1, 0.25), vec(0, 0, 1)        # A and A2 = same person


def fm(name, score, q, emb):
    return FaceMatch(name, score, b"jpg", q, 0.3, emb)


def test_summarize_votes_per_person_and_prefers_best_frame():
    groups = summarize([fm("Sam", 0.40, 0.3, A), fm(None, 0.1, 0.8, A2),
                        fm("Sam", 0.55, 0.7, A), fm(None, 0.05, 0.6, B)], ARCFACE.group)
    assert [g.name for g in groups] == ["Sam", None]
    assert groups[0].score == 0.55 and groups[0].best.quality == 0.8 and len(groups[0].members) == 3


def test_summarize_conflicting_names_need_clear_winner():
    tied = summarize([fm("Sam", 0.40, 0.5, A), fm("Alex", 0.38, 0.5, A2)], ARCFACE.group)
    assert len(tied) == 1 and tied[0].name is None
    clear = summarize([fm("Sam", 0.60, 0.9, A), fm("Sam", 0.5, 0.8, A2),
                       fm("Alex", 0.27, 0.2, A2)], ARCFACE.group)
    assert clear[0].name == "Sam"


class FakeRec:
    backend = ARCFACE

    def __init__(self):
        self.known: dict[str, list[np.ndarray]] = {}
        self.added: list[str] = []

    def embed(self, img):
        return A

    def identify(self, emb):
        best = max(((n, float(e @ emb)) for n, es in self.known.items() for e in es),
                   key=lambda t: t[1], default=(None, 0.0))
        return (best[0] if best[1] >= self.backend.match else None), best[1], best[1]

    def suggest(self, embs):
        return None

    def add(self, name, jpg, reload=True):
        self.added.append(name)

    def reload(self):
        self.known.setdefault("Sam", []).append(A)


def test_unknown_groups_assign_and_resolve(tmp_path):
    rec = FakeRec()
    store = UnknownFaces(tmp_path / "u", rec)
    now = datetime.now().astimezone()
    store.add(fm(None, 0.1, 0.9, A), "deurbel", now, "deurbel/d/1_person.face1.jpg")
    store.add(fm(None, 0.1, 0.5, A2), "hek", now, "hek/d/2_person.face1.jpg")
    store.add(fm(None, 0.1, 0.6, B), "deurbel", now, "deurbel/d/3_person.face1.jpg")
    assert not store.add(fm(None, 0.1, 0.1, B), "deurbel", now, "x.face1.jpg")   # too poor
    groups = store.groups()
    assert [g["count"] for g in groups] == [2, 1] and groups[0]["cameras"] == ["deurbel", "hek"]
    # only the first of the group gets a name; A2 is then recognised automatically
    updates = store.assign(groups[0]["ids"][:1], "Sam")
    assert sorted(updates) == [("deurbel/d/1_person.face1.jpg", "Sam"),
                               ("hek/d/2_person.face1.jpg", "Sam")]
    assert rec.added == ["Sam"] and store.count() == 1
    assert UnknownFaces(tmp_path / "u", rec).count() == 1          # kept after a restart
    assert store.dismiss([g for g in store.groups()[0]["ids"]]) == 1 and store.count() == 0


def test_unknown_prunes_old(tmp_path):
    store = UnknownFaces(tmp_path / "u", FakeRec())
    store.add(fm(None, 0.1, 0.9, A), "hek", datetime.now().astimezone() - timedelta(days=40), "a.face1.jpg")
    assert store.count() == 0


def test_event_face_name_updated_on_disk(tmp_path):
    day = tmp_path / "deurbel" / "2026-11-02"
    day.mkdir(parents=True)
    crop = "deurbel/2026-11-02/101010_person.face1.jpg"
    (day / "101010_person.json").write_text(json.dumps(
        {"camera": "deurbel", "faces": [{"crop": crop, "name": None, "score": 0.1}]}))
    store = EventStore(tmp_path, "http://x")
    assert store.set_face_name(crop, "Sam")
    assert json.loads((day / "101010_person.json").read_text())["faces"][0]["name"] == "Sam"
    assert not store.set_face_name("deurbel/2026-11-02/nope.face1.jpg", "Sam")
    other = "deurbel/2026-11-02/101010_person.face2.jpg"
    data = json.loads((day / "101010_person.json").read_text())
    data["faces"].append({"crop": other, "name": None})
    (day / "101010_person.json").write_text(json.dumps(data))
    assert store.remove_face(crop) == (True, "Sam")
    assert [f["crop"] for f in json.loads((day / "101010_person.json").read_text())["faces"]] == [other]
    assert store.remove_face(other) == (True, None)
    assert json.loads((day / "101010_person.json").read_text())["faces"] is None
    assert store.remove_face(other) == (False, None)


def test_face_answer_from_notification(tmp_path):
    from types import SimpleNamespace
    from camsvc.service import Service

    rec = FakeRec()
    rec.unknown = UnknownFaces(tmp_path / "u", rec)
    now = datetime.now().astimezone()
    sid = rec.unknown.add(fm(None, 0.1, 0.9, A), "deurbel", now, "deurbel/d/1_person.face1.jpg")
    rec.unknown.add(fm(None, 0.1, 0.8, A2), "deurbel", now, "deurbel/d/2_person.face1.jpg")
    other = rec.unknown.add(fm(None, 0.1, 0.8, B), "hek", now, "hek/d/3_person.face1.jpg")
    named = []
    svc = SimpleNamespace(faces=rec, store=SimpleNamespace(set_face_name=lambda c, n: named.append((c, n))))
    Service._on_face_answer(svc, sid, "Sam", False)
    assert sorted(named) == [("deurbel/d/1_person.face1.jpg", "Sam"), ("deurbel/d/2_person.face1.jpg", "Sam")]
    Service._on_face_answer(svc, other, None, True)
    assert rec.unknown.count() == 0
    Service._on_face_answer(svc, "000000000000", "Sam", False)     # already handled: no error


def test_name_needs_most_frames_or_the_sharpest():
    from camsvc.faces import FaceMatch, summarize
    import numpy as np
    e = np.array([1.0, 0, 0], np.float32)
    m = lambda name, score, q: FaceMatch(name, score, b"", q, 0.1, e)
    # stranger: sharpest frame without a name, 3 of 9 frames resemble Alex
    stranger = [m(None, 0.17, 0.55)] + [m("Alex", 0.35, 0.3)] * 3 + [m(None, 0.1, 0.2)] * 5
    assert summarize(stranger, 0.3)[0].name is None
    # known person: the sharpest frame carries the name, even if most frames are doubtful
    sam = [m("Sam", 0.35, 0.78)] + [m("Sam", 0.3, 0.4)] * 2 + [m(None, 0.2, 0.3)] * 4
    assert summarize(sam, 0.3)[0].name == "Sam"
    # known person: the sharpest frame is doubtful, but most frames match
    alex = [m(None, 0.38, 0.5)] + [m("Alex", 0.45, 0.4)] * 11 + [m(None, 0.2, 0.3)] * 2
    assert summarize(alex, 0.3)[0].name == "Alex"


def test_learn_auto_skips_weak_embeddings(tmp_path):
    import threading
    import numpy as np
    from camsvc.faces import ARCFACE, FaceGroup, FaceMatch, FaceRecognizer
    r = FaceRecognizer.__new__(FaceRecognizer)
    r.backend, r._lock, r.faces_dir = ARCFACE, threading.Lock(), tmp_path
    r._names, r._mat = [], np.zeros((0, 3), np.float32)
    r.reload = lambda: None
    e = np.array([1.0, 0, 0], np.float32)
    import cv2
    color = cv2.imencode(".jpg", np.full((16, 16, 3), (80, 110, 170), np.uint8))[1].tobytes()
    gray = cv2.imencode(".jpg", np.full((16, 16, 3), 120, np.uint8))[1].tobytes()
    wood = FaceMatch("Sam", 0.6, color, 0.6, 0.3, e, norm=12.9)   # a piece of wood that resembled Sam
    assert not r.learn_auto(FaceGroup("Sam", 0.6, wood, [wood]))
    night = FaceMatch("Sam", 0.6, gray, 0.6, 0.3, e, norm=21.0)   # do not learn from infrared
    assert not r.learn_auto(FaceGroup("Sam", 0.6, night, [night]))
    face = FaceMatch("Sam", 0.6, color, 0.6, 0.3, e, norm=21.0)
    assert r.learn_auto(FaceGroup("Sam", 0.6, face, [face]))
    assert len(list((tmp_path / "Sam" / "auto").glob("*.jpg"))) == 1


def test_confirmed_weak_needs_a_run():
    import numpy as np
    from camsvc.faces import FaceMatch, confirmed_weak
    e = np.array([1.0, 0, 0], np.float32)
    m = lambda name: FaceMatch(name, 0.5, b"", 0.1, 0.3, e, 21.0, True)
    assert confirmed_weak([(1.0, m("Sam")), (1.5, m("Sam"))]) == []
    assert len(confirmed_weak([(1.0, m("Sam")), (1.5, m("Sam")), (2.0, m("Sam"))])) == 3
    assert confirmed_weak([(1.0, m("Sam")), (1.5, m("Noor")), (2.0, m("Sam"))]) == []
    assert confirmed_weak([(1.0, m("Sam")), (5.0, m("Sam")), (9.0, m("Sam"))]) == []


def test_forget_like_removes_only_near_duplicates(tmp_path):
    import threading
    import cv2
    import numpy as np
    from camsvc.faces import FaceRecognizer
    r = FaceRecognizer.__new__(FaceRecognizer)
    r._lock, r.faces_dir = threading.Lock(), tmp_path
    r.reload = lambda: None
    r._raw_embed = lambda img: img.reshape(-1, 3).mean(0).astype(np.float32)   # colour as the 'face'
    jpg = lambda bgr: cv2.imencode(".jpg", np.full((16, 16, 3), bgr, np.uint8))[1].tobytes()
    (tmp_path / "Sam" / "auto").mkdir(parents=True)
    (tmp_path / "Sam" / "auto" / "aaaaaaaaaaaa.jpg").write_bytes(jpg((200, 40, 40)))   # same image
    (tmp_path / "Sam" / "bbbbbbbbbbbb.jpg").write_bytes(jpg((40, 40, 200)))            # another image
    (tmp_path / "Noor").mkdir()
    (tmp_path / "Noor" / "cccccccccccc.jpg").write_bytes(jpg((200, 40, 40)))           # another person
    assert r.forget_like("Sam", jpg((200, 40, 40))) == 1
    assert not (tmp_path / "Sam" / "auto").exists()
    assert (tmp_path / "Sam" / "bbbbbbbbbbbb.jpg").exists() and (tmp_path / "Noor" / "cccccccccccc.jpg").exists()
