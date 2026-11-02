"""AI-beschrijvingen: contract en foutpaden tegen een nep-Ollama-server (geen echte inferentie)."""

import base64
import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from camsvc.ai_describe import (DEFAULTS, IR_COLOR_NOTE, AiDescriber, AiError, AiJob,
                                false_alarm, is_grayscale, prepare_image, validate_output)
from camsvc.config import _load_ai

GOOD = {"beschrijving": "Een persoon staat bij de deur.", "objecten": ["persoon", "deur"],
        "kenmerken": ["blauwe jas", "pakket"], "beeldkwaliteit": "goed", "onzekerheden": [],
        "persoon_zichtbaar": True, "voertuig_zichtbaar": False, "dier_zichtbaar": False}


class FakeOllama:
    """Antwoordt met een vooraf ingestelde reeks (status, body, headers); bewaart verzoeken."""

    def __init__(self):
        self.responses: list[tuple[int, object, dict]] = []
        self.requests: list[dict] = []
        self.models = [DEFAULTS["model"]]
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, headers=None):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200, {"models": [{"name": m} for m in fake.models]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(body)
                code, resp, headers = fake.responses.pop(0) if fake.responses else (500, {}, {})
                self._send(code, resp, headers)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def chat_ok(self, content=GOOD, done_reason="stop"):
        text = content if isinstance(content, str) else json.dumps(content)
        self.responses.append((200, {"done": True, "done_reason": done_reason,
                                     "message": {"role": "assistant", "content": text}}, {}))


@pytest.fixture
def fake():
    f = FakeOllama()
    yield f
    f.server.shutdown()


def make(fake, **over):
    cfg = {**DEFAULTS, "enabled": True, "base_url": fake.url, **over}
    results = []
    d = AiDescriber(cfg, lambda cam, ev, res: results.append((cam, ev, res)))
    return d, results


def job(event_id="e1", camera="deurbel", image=None, age=0.0, now=None):
    img = image if image is not None else np.random.default_rng(1).integers(
        0, 255, (360, 640, 3), dtype=np.uint8)
    import time
    return AiJob(event_id, camera, "persoon", datetime.now().astimezone(),
                 (now or time.monotonic()) - age, img, [])


def color_img():
    img = np.zeros((360, 640, 3), np.uint8)
    img[:, :, 2] = 200                                  # duidelijk rood
    return img


# ---------------------------------------------------------------- contract
def test_success_payload_and_parsing(fake):
    fake.chat_ok()
    d, results = make(fake)
    d._process(job(image=color_img()))
    req = fake.requests[0]
    assert req["model"] == DEFAULTS["model"] and req["stream"] is False and req["format"]["required"]
    img_b64 = req["messages"][1]["images"][0]
    assert not img_b64.startswith("data:") and base64.b64decode(img_b64)[:2] == b"\xff\xd8"
    assert "infraroodbeeld" not in req["messages"][1]["content"]   # kleurbeeld: geen IR-hint
    cam, ev, res = results[0]
    assert (cam, ev, res["status"]) == ("deurbel", "e1", "ok")
    assert res["beschrijving"] == GOOD["beschrijving"] and res["model"] == DEFAULTS["model"]
    assert res["prompt_versie"] and res["opgenomen"] and "duur_ms" in res


def test_infrared_hint_and_color_flag(fake):
    fake.chat_ok({**GOOD, "beschrijving": "Een zwarte auto onder de carport."})
    d, results = make(fake)
    gray = np.full((360, 640, 3), 90, np.uint8)
    d._process(job(image=gray))
    assert "infraroodbeeld" in fake.requests[0]["messages"][1]["content"]
    res = results[0][2]
    assert res["infrarood"] and res["kleur_onbetrouwbaar"] and IR_COLOR_NOTE in res["onzekerheden"]


@pytest.mark.parametrize("bad", [
    {**GOOD, "extra": 1},
    {**GOOD, "beeldkwaliteit": "uitstekend"},
    {**GOOD, "persoon_zichtbaar": "ja"},
    {**GOOD, "beschrijving": "x" * 501},
    {**GOOD, "objecten": ["a"] * 9},
    {k: v for k, v in GOOD.items() if k != "onzekerheden"},
    "geen json",
])
def test_invalid_output_rejected_without_retry(fake, bad):
    fake.chat_ok(bad)
    d, results = make(fake)
    d._process(job())
    assert results[0][2]["status"] == "invalid_response" and len(fake.requests) == 1


def test_truncated_answer_rejected(fake):
    fake.chat_ok(done_reason="length")
    d, results = make(fake)
    d._process(job())
    assert results[0][2] == {**results[0][2], "status": "invalid_response", "reden": "truncated"}


# ---------------------------------------------------------------- fouten en retries
def test_one_retry_on_503_then_ok(fake):
    fake.responses.append((503, {}, {}))
    fake.chat_ok()
    d, results = make(fake)
    d._process(job())
    assert results[0][2]["status"] == "ok" and len(fake.requests) == 2


def test_no_retry_on_missing_model(fake):
    fake.responses.append((404, {"error": "model not found"}, {}))
    d, results = make(fake)
    d._process(job())
    assert results[0][2]["reden"] == "model_missing" and len(fake.requests) == 1
    assert d.state == "model_missing" and not d.wants("hek", "persoon")


def test_unreachable_server_reports_unavailable():
    cfg = {**DEFAULTS, "enabled": True, "base_url": "http://127.0.0.1:1", "max_retries": 0}
    results = []
    d = AiDescriber(cfg, lambda *a: results.append(a))
    d._process(job())
    assert results[0][2]["status"] == "unavailable" and results[0][2]["reden"] == "connection"


def test_breaker_opens_after_failures(fake):
    d, results = make(fake, max_retries=0, breaker_failures=3, breaker_cooldown=60)
    for i in range(3):
        fake.responses.append((503, {}, {}))
        d._process(job(event_id=f"e{i}"))
    assert d.state == "breaker"
    d._process(job(event_id="e9"))
    assert results[-1][2] == {**results[-1][2], "status": "unavailable", "reden": "breaker"}
    assert len(fake.requests) == 3                      # geen nieuwe aanroep tijdens de pauze


# ---------------------------------------------------------------- wachtrij en limieten
def test_queue_replaces_waiting_image_per_camera(fake):
    d, results = make(fake)
    d.submit(job(event_id="oud"))
    d.submit(job(event_id="nieuw"))
    assert results[0][1] == "oud" and results[0][2]["reden"] == "vervangen"
    assert list(d._pending) == ["deurbel"] and d._pending["deurbel"].event_id == "nieuw"


def test_camera_interval_and_labels(fake):
    d, _ = make(fake, camera_interval=30)
    assert d.wants("hek", "persoon") and not d.wants("hek", "beweging")
    d.submit(job(camera="hek"))
    assert not d.wants("hek", "persoon") and d.wants("carport", "voertuig")


def test_old_event_expires_without_call(fake):
    d, results = make(fake, event_max_age=120)
    d._process(job(age=200))
    assert results[0][2]["status"] == "expired" and not fake.requests


def test_global_rate_limit_counts_starts(fake):
    d, _ = make(fake, global_per_minute=2)
    import time
    deadline = time.monotonic() + 0.2
    assert d._wait_for_slot(deadline) and d._wait_for_slot(deadline)
    assert not d._wait_for_slot(deadline)               # derde start past niet binnen de deadline


# ---------------------------------------------------------------- beeld en config
def test_prepare_image_masks_resizes_and_limits():
    img = np.full((2160, 3840, 3), 200, np.uint8)
    jpg, gray = prepare_image(img, [[[0, 0], [100, 0], [100, 50], [0, 50]]], 1280, 2 * 1024 * 1024,
                              20_000_000)
    import cv2
    out = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert max(out.shape[:2]) == 1280 and gray
    assert out[:300].mean() < 5 and out[500:].mean() > 150   # bovenste helft zwart gemaskeerd
    with pytest.raises(AiError):
        prepare_image(np.zeros((5000, 5000, 3), np.uint8), [], 1280, 2 ** 21, 20_000_000)
    with pytest.raises(AiError):
        prepare_image(np.zeros((10, 10, 3), np.uint8), [], 1280, 2 ** 21, 20_000_000)


def test_crop_around_scales_box_from_substream():
    from camsvc.ai_describe import crop_around
    main = np.zeros((1440, 2560, 3), np.uint8)
    crop = crop_around(main, (300, 100, 340, 200), (360, 640))   # box in sub-stream (640x360)
    # 4x opgeschaald: 160x400 px, x1.3 marge -> vierkant van 520 px
    assert crop.shape[:2] == (520, 520)
    assert crop_around(main, None, (360, 640)) is main
    edge = crop_around(main, (0, 0, 20, 20), (360, 640))         # aan de rand: wordt afgekapt
    assert edge.shape[0] <= 520 and edge.shape[1] <= 520


def test_grayscale_detection():
    assert is_grayscale(np.full((100, 100, 3), 77, np.uint8))
    assert not is_grayscale(color_img())


def test_config_defaults_off_and_loopback_only():
    assert _load_ai(None)["enabled"] is False
    with pytest.raises(ValueError):
        _load_ai({"base_url": "http://192.168.1.50:11434"})
    with pytest.raises(ValueError):
        _load_ai({"queue_max": -1})
    with pytest.raises(ValueError):
        _load_ai({"onbekend": 1})


def test_whole_sentences_cuts_broken_text():
    from camsvc.ai_describe import whole_sentences
    assert whole_sentences("Een auto staat op de oprit. De auto is gedeeltelijk bedekt waardoor de vorm on") \
        == "Een auto staat op de oprit."
    assert whole_sentences("Een persoon loopt langs.") == "Een persoon loopt langs."
    assert whole_sentences("Geen punt maar ook geen tweede zin") == "Geen punt maar ook geen tweede zin"


def test_num_gpu_passed_only_when_set(fake):
    fake.chat_ok(); fake.chat_ok()
    d, _ = make(fake)
    d._process(job(event_id="a"))
    assert "num_gpu" not in fake.requests[0]["options"]
    d2, _ = make(fake, num_gpu=37)
    d2._process(job(event_id="b"))
    assert fake.requests[1]["options"]["num_gpu"] == 37
    with pytest.raises(ValueError):
        _load_ai({"num_gpu": 0})


def test_validate_output_strips_and_keeps_contract():
    out = validate_output(json.dumps({**GOOD, "objecten": [" persoon ", ""]}), False)
    assert out["objecten"] == ["persoon"] and out["infrarood"] is False


def test_false_alarm_verdict():
    ok = {"status": "ok", "beeldkwaliteit": "goed", "infrarood": False,
          "persoon_zichtbaar": False, "voertuig_zichtbaar": False, "dier_zichtbaar": False}
    assert false_alarm(["persoon"], ok)                                   # plant bij het hek
    assert not false_alarm(["persoon"], {**ok, "persoon_zichtbaar": True})
    assert not false_alarm(["aanbellen", "persoon"], ok)                  # aanbellen nooit vals
    assert not false_alarm(["persoon", "voertuig"], {**ok, "voertuig_zichtbaar": True})
    assert false_alarm(["persoon"], {**ok, "beeldkwaliteit": "onbruikbaar"})       # overdag telt het niet
    assert not false_alarm(["persoon"], {**ok, "infrarood": True, "beeldkwaliteit": "onbruikbaar"})
    assert not false_alarm(["persoon"], {**ok, "infrarood": True, "beeldkwaliteit": "beperkt"})
    assert false_alarm(["persoon"], {**ok, "infrarood": True})
    assert not false_alarm(["persoon"], {"status": "ok", "beeldkwaliteit": "goed"})  # oud resultaat
    assert not false_alarm(["beweging"], ok) and not false_alarm(["persoon"], None)
