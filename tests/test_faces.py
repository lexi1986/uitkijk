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
