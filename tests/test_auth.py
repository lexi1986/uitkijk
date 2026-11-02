"""Passkey login: sessions, pairing code and access control (without a real WebAuthn ceremony)."""

import time

import pytest

from camsvc.auth import Auth, AuthError
from camsvc.config import _load_auth


def test_auth_config_requires_https_domain():
    assert _load_auth(None)["enabled"] is False
    ok = _load_auth({"enabled": True, "rp_id": "camera.example.nl", "origin": "https://camera.example.nl/"})
    assert ok["origin"] == "https://camera.example.nl" and ok["session_days"] == 90
    for bad in ({"enabled": True, "rp_id": "camera.example.nl", "origin": "http://camera.example.nl"},
                {"enabled": True, "rp_id": "a.nl", "origin": "https://b.nl"}, {"onzin": 1}):
        with pytest.raises(ValueError):
            _load_auth(bad)


def test_setup_code_sessions_and_logout(tmp_path):
    auth = Auth(tmp_path, "camera.example.nl", "https://camera.example.nl", session_days=1)
    code = (tmp_path / "pairing_code").read_text().strip()
    with pytest.raises(AuthError):
        auth.registration_options(logged_in=False, setup_code="FOUT-FOUT-FOUT")
    opts = auth.registration_options(logged_in=False, setup_code=code.lower())
    assert opts["options"]["rp"]["id"] == "camera.example.nl" and opts["challenge_id"]
    # simulate a registered passkey
    data = auth._load()
    data["credentials"] = [{"id": "abc", "public_key": "x", "sign_count": 0, "name": "Mac",
                            "created": "2026-11-02", "last_used": "2026-11-02"}]
    auth._save(data)
    cookie = auth.make_cookie("abc")
    assert auth.session_ok(cookie) and not auth.session_ok(cookie + "0") and not auth.session_ok(None)
    auth.logout_everywhere()
    assert not auth.session_ok(cookie)
    cookie = auth.make_cookie("abc")
    assert auth.remove("abc") and not auth.session_ok(cookie)        # passkey gone = logged out
    assert (tmp_path / "pairing_code").exists()                       # and a pairing code again


def test_challenge_single_use(tmp_path):
    auth = Auth(tmp_path, "camera.example.nl", "https://camera.example.nl")
    cid = auth.login_options()["challenge_id"]
    with pytest.raises(AuthError):
        auth.login(cid, {"id": "unknown"})
    with pytest.raises(AuthError):
        auth.login(cid, {"id": "unknown"})                          # already used


def test_setup_code_survives_restart(tmp_path):
    Auth(tmp_path, "camera.example.nl", "https://camera.example.nl")
    code = (tmp_path / "pairing_code").read_text()
    Auth(tmp_path, "camera.example.nl", "https://camera.example.nl")
    assert (tmp_path / "pairing_code").read_text() == code
