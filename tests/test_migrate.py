"""Migration of data written by the earlier Dutch version."""

import json

from camsvc.config import load_config
from camsvc.events import EventRecord
from camsvc.migrate import migrate


def test_migrates_clips_state_and_config(tmp_path):
    clips, state = tmp_path / "clips", tmp_path / "state"
    day = clips / "hek" / "2026-11-02"
    day.mkdir(parents=True)
    state.mkdir()
    for ext in ("mp4", "jpg", "face1.jpg"):
        (day / f"100000_persoon.{ext}").write_bytes(b"x")
    (day / "100000_persoon.json").write_text(json.dumps({
        "id": "e1", "camera": "hek", "label": "persoon", "labels": ["persoon", "voertuig"], "score": 0.9,
        "start": "2026-11-02T10:00:00+01:00", "end": "", "duration": 30,
        "clip": "hek/2026-11-02/100000_persoon.mp4", "snapshot": "hek/2026-11-02/100000_persoon.jpg",
        "faces": [{"crop": "hek/2026-11-02/100000_persoon.face1.jpg", "name": None, "score": 0.1}],
        "vals": "gebruiker", "vermoedelijk": ["Alex"],
        "extern": {"melden": True, "titel": "Bezoeker", "beschrijving": "Iemand", "tijd": "t"},
        "ai": {"status": "ok", "beschrijving": "Een plant.", "objecten": [], "kenmerken": [],
               "beeldkwaliteit": "onbruikbaar", "onzekerheden": [], "infrarood": False,
               "persoon_zichtbaar": False, "duur_ms": 1000}}))
    (state / "koppelcode").write_text("AAAA-BBBB-CCCC\n")
    (state / "kleding.json").write_text("{}")
    (state / "vals_hek.json").write_text(json.dumps([{"label": "persoon", "box": [0, 0, 1, 1], "event": "e1"}]))
    (state / "stationary_hek.json").write_text(json.dumps([{"label": "voertuig", "box": [1, 2, 3, 4], "age": 1}]))
    (state / "settings.json").write_text(json.dumps({"hek": {"objects.labels.dier": True, "pre_seconds": 5}}))
    (state / "auth.json").write_text(json.dumps({"secret": "s", "credentials": [
        {"id": "c", "naam": "iPhone", "aangemaakt": "2026-11-02", "laatst": "2026-11-02"}]}))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"""storage:
  clips_dir: {clips}
  state_dir: {state}
  retention_days_by_label: {{persoon: 30, aanbellen: 30}}
defaults:
  objects:
    labels:
      persoon: true
      dier: true
cameras:
  - name: hek
    main_url: rtsp://user:persoon@10.0.0.2/stream   # a password that looks like a label stays
""")
    assert migrate(cfg, clips, state)
    assert not migrate(cfg, clips, state)                                 # only once
    names = sorted(p.name for p in day.iterdir())
    assert names == ["100000_person.face1.jpg", "100000_person.jpg", "100000_person.json", "100000_person.mp4"]
    ev = json.loads((day / "100000_person.json").read_text())
    rec = EventRecord(**ev)
    assert rec.label == "person" and rec.labels == ["person", "vehicle"] and rec.false_alarm == "user"
    assert rec.likely == ["Alex"] and rec.external == {"notify": True, "title": "Bezoeker", "description": "Iemand", "time": "t"}
    assert rec.clip.endswith("100000_person.mp4") and rec.faces[0]["crop"].endswith("100000_person.face1.jpg")
    assert rec.ai["image_quality"] == "unusable" and rec.ai["person_visible"] is False and rec.ai["duration_ms"] == 1000
    assert (state / "pairing_code").exists() and (state / "clothing.json").exists()
    assert json.loads((state / "false_spots_hek.json").read_text())[0]["label"] == "person"
    assert json.loads((state / "stationary_hek.json").read_text())[0]["label"] == "vehicle"
    assert json.loads((state / "settings.json").read_text()) == {"hek": {"objects.labels.animal": True, "pre_seconds": 5}}
    assert json.loads((state / "auth.json").read_text())["credentials"][0]["name"] == "iPhone"
    loaded = load_config(cfg)
    assert loaded.retention_days_by_label == {"person": 30.0, "doorbell": 30.0}
    assert set(loaded.cameras[0].enabled_labels()) == {"person", "vehicle", "animal"}
    assert loaded.cameras[0].main_url == "rtsp://user:persoon@10.0.0.2/stream"
    assert (tmp_path / "config.yaml.before-english").exists()


def test_fresh_install_is_left_alone(tmp_path):
    (tmp_path / "state").mkdir()
    assert not migrate(tmp_path / "config.yaml", tmp_path / "clips", tmp_path / "state")
    assert (tmp_path / "state" / ".migrated-english").exists()
