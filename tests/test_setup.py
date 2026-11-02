"""Setup wizard: from answers to a config the service can really load."""

import pytest
import yaml
from fastapi.testclient import TestClient

from camsvc.config import load_config
from camsvc.setup import build_config, camera_urls, create_setup_app, go2rtc_yaml, slug


def test_brand_urls_encode_credentials():
    main, sub = camera_urls({"brand": "reolink", "ip": "192.168.1.30", "username": "admin", "password": "p@ss:w/rd!"})
    assert main == "rtsp://admin:p%40ss%3Aw%2Frd%21@192.168.1.30:554/h264Preview_01_main"
    assert sub.endswith("/h264Preview_01_sub")
    assert camera_urls({"brand": "foscam", "ip": "10.0.0.2"})[0] == "rtsp://10.0.0.2:88/videoMain"
    assert camera_urls({"brand": "anders", "main_url": "rtsp://x/1", "sub_url": ""}) == ("rtsp://x/1", None)
    with pytest.raises(ValueError):
        camera_urls({"brand": "anders", "name": "x"})


def test_build_config_loads(tmp_path):
    cfg = build_config({
        "cameras": [{"friendly_name": "Voordeur", "brand": "foscam", "ip": "10.0.0.2", "username": "u", "password": "p"},
                    {"friendly_name": "Oprit 2", "brand": "reolink", "ip": "10.0.0.3", "plates": True}],
        "mqtt": {"host": "10.0.0.5", "username": "cam", "password": "geheim"},
        "features": {"faces": True, "plates": True, "live": True},
        "storage": {"days": 5, "person_days": 20, "max_gb": 40},
        "public_url": "http://10.0.0.9:8080/", "state_dir": str(tmp_path / "state"), "clips_dir": str(tmp_path / "clips"),
    })
    f = tmp_path / "config.yaml"
    f.write_text(yaml.safe_dump(cfg))
    loaded = load_config(f)
    names = [c.name for c in loaded.cameras]
    assert names == ["voordeur", "oprit_2"]
    assert loaded.cameras[0].live_mode == "webrtc,mse" and loaded.cameras[0].settings["faces"]["enabled"]
    assert loaded.cameras[1].settings["plates"]["enabled"] and not loaded.cameras[0].settings["plates"]["enabled"]
    assert loaded.mqtt["host"] == "10.0.0.5" and loaded.go2rtc_url == "http://localhost:1984"
    assert loaded.retention_days == 5 and loaded.public_url == "http://10.0.0.9:8080"
    assert "voordeur:" in go2rtc_yaml(cfg)


def test_build_config_rejects_bad_input():
    with pytest.raises(ValueError):
        build_config({"cameras": []})
    with pytest.raises(ValueError):
        build_config({"cameras": [{"friendly_name": "A", "brand": "tapo", "ip": "1.2.3.4"},
                                  {"friendly_name": "a", "brand": "tapo", "ip": "1.2.3.5"}]})
    with pytest.raises(ValueError):
        build_config({"cameras": [{"friendly_name": "A", "brand": "tapo", "ip": "1.2.3.4"}], "domain": "not a domain"})
    auth = build_config({"cameras": [{"friendly_name": "A", "brand": "tapo", "ip": "1.2.3.4"}], "domain": "cam.voorbeeld.nl"})
    assert auth["auth"]["origin"] == "https://cam.voorbeeld.nl" and auth["api"]["public_url"] == "https://cam.voorbeeld.nl"
    assert build_config({"cameras": [{"friendly_name": "A", "brand": "tapo", "ip": "1.2.3.4"}]})["mqtt"]["host"] == ""
    assert slug("Tuin achter!") == "tuin_achter"


def test_setup_api_needs_code(tmp_path, monkeypatch):
    app = create_setup_app(tmp_path / "config.yaml", tmp_path / "state", "test")
    client = TestClient(app)
    code = (tmp_path / "state" / "setup_code").read_text().strip()
    assert client.get("/").status_code == 200
    assert client.post("/api/setup/check").status_code == 401
    assert client.post("/api/setup/check", headers={"X-Setup-Code": code.lower()}).status_code == 200
    exits = []
    monkeypatch.setattr("camsvc.setup.threading.Timer", lambda *a: type("T", (), {"start": lambda s: exits.append(1)})())
    r = client.post("/api/setup/save", headers={"X-Setup-Code": code},
                    json={"cameras": [{"friendly_name": "Hek", "brand": "anders", "main_url": "rtsp://x/1"}],
                          "state_dir": str(tmp_path / "state")})
    assert r.status_code == 200 and exits == [1]
    assert load_config(tmp_path / "config.yaml").cameras[0].name == "hek"
    assert not (tmp_path / "state" / "setup_code").exists()
    assert (tmp_path / "config.yaml").stat().st_mode & 0o077 == 0
