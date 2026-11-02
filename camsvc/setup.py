"""Setup wizard: without a (usable) config the service starts only this page.

- Protected by a one-time setup code printed in the log (`docker compose logs`), so not everyone
  on the network can do the first setup.
- Cameras per brand (RTSP addresses are filled in) with a test button that really fetches a frame.
- Saving writes config.yaml (and the streams for go2rtc) and stops the process; Docker or systemd
  then starts it again, now in normal mode.
"""

from __future__ import annotations

import hmac
import logging
import os
import re
import secrets
import threading
import urllib.parse
import urllib.request
from pathlib import Path

import yaml
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
TOKEN_FILE = "setup_code"

# RTSP addresses per brand: (main, sub). {auth} = user:password@ (URL-encoded).
BRANDS: dict[str, dict] = {
    "reolink": {"name": "Reolink", "port": 554,
                "main": "rtsp://{auth}{ip}:{port}/h264Preview_01_main",
                "sub": "rtsp://{auth}{ip}:{port}/h264Preview_01_sub"},
    "foscam": {"name": "Foscam", "port": 88,
               "main": "rtsp://{auth}{ip}:{port}/videoMain", "sub": "rtsp://{auth}{ip}:{port}/videoSub",
               "live_mode": "webrtc,mse"},
    "hikvision": {"name": "Hikvision / Annke", "port": 554,
                  "main": "rtsp://{auth}{ip}:{port}/Streaming/Channels/101",
                  "sub": "rtsp://{auth}{ip}:{port}/Streaming/Channels/102"},
    "dahua": {"name": "Dahua / Amcrest / Lorex", "port": 554,
              "main": "rtsp://{auth}{ip}:{port}/cam/realmonitor?channel=1&subtype=0",
              "sub": "rtsp://{auth}{ip}:{port}/cam/realmonitor?channel=1&subtype=1"},
    "tapo": {"name": "TP-Link Tapo / VIGI", "port": 554,
             "main": "rtsp://{auth}{ip}:{port}/stream1", "sub": "rtsp://{auth}{ip}:{port}/stream2"},
    "unifi": {"name": "UniFi Protect (RTSP link from Protect)", "port": 7447, "main": "", "sub": ""},
    "other": {"name": "Other (own RTSP address)", "port": 554, "main": "", "sub": ""},
}
_SLUG = re.compile(r"[^a-z0-9_]+")


def slug(text: str) -> str:
    s = _SLUG.sub("_", text.strip().lower()).strip("_")
    return s or "camera"


def camera_urls(cam: dict) -> tuple[str, str | None]:
    """Main and sub address for a camera from the wizard (brand + IP, or own addresses)."""
    brand = BRANDS.get(cam.get("brand") or "other", BRANDS["other"])
    if cam.get("main_url"):
        return str(cam["main_url"]).strip(), (str(cam.get("sub_url") or "").strip() or None)
    if not brand["main"] or not cam.get("ip"):
        raise ValueError(f"Camera '{cam.get('name', '?')}': enter the IP address or an RTSP address")
    user, pw = str(cam.get("username") or ""), str(cam.get("password") or "")
    auth = (f"{urllib.parse.quote(user, safe='')}:{urllib.parse.quote(pw, safe='')}@" if user else "")
    fields = {"auth": auth, "ip": str(cam["ip"]).strip(), "port": int(cam.get("port") or brand["port"])}
    return brand["main"].format(**fields), brand["sub"].format(**fields)


def build_config(data: dict) -> dict:
    """Config (as a dict) from the wizard answers; raises ValueError on errors."""
    cams = data.get("cameras") or []
    if not cams:
        raise ValueError("Add at least one camera")
    features = data.get("features") or {}
    out_cams, seen = [], set()
    for cam in cams:
        friendly = str(cam.get("friendly_name") or cam.get("name") or "").strip()
        if not friendly:
            raise ValueError("Every camera needs a name")
        name = slug(cam.get("name") or friendly)
        if name in seen:
            raise ValueError(f"The name '{friendly}' is used twice")
        seen.add(name)
        main, sub = camera_urls(cam)
        entry: dict = {"name": name, "friendly_name": friendly, "main_url": main}
        if sub:
            entry["sub_url"] = sub
        live_mode = BRANDS.get(cam.get("brand") or "", {}).get("live_mode")
        if live_mode:
            entry["live_mode"] = live_mode
        if features.get("faces"):
            entry["faces"] = {"enabled": True}
        if features.get("plates") and cam.get("plates"):
            entry["plates"] = {"enabled": True}
        out_cams.append(entry)

    mqtt_in = data.get("mqtt") or {}
    storage = data.get("storage") or {}
    cfg: dict = {
        "log_level": "INFO",
        "storage": {"clips_dir": str(data.get("clips_dir") or "/clips"),
                    "state_dir": str(data.get("state_dir") or "/data/state"),
                    "retention_days": float(storage.get("days") or 7),
                    "retention_days_by_label": {"person": float(storage.get("person_days") or 30),
                                                "doorbell": float(storage.get("person_days") or 30)},
                    "max_gb": float(storage.get("max_gb") or 50)},
        "api": {"host": "0.0.0.0", "port": int(data.get("port") or 8080),
                "public_url": str(data.get("public_url") or "http://localhost:8080").rstrip("/")},
        "model": {"path": str(data.get("model_path") or "/data/models/yolo11n.onnx"), "input_size": 320},
        "summary_time": "22:00" if mqtt_in.get("host") else "",
        "cameras": out_cams,
    }
    if mqtt_in.get("host"):
        cfg["mqtt"] = {"host": str(mqtt_in["host"]).strip(), "port": int(mqtt_in.get("port") or 1883),
                       "username": str(mqtt_in.get("username") or "") or None,
                       "password": str(mqtt_in.get("password") or "") or None}
    else:
        cfg["mqtt"] = {"host": "", "port": 1883}
    if features.get("live"):
        cfg["live"] = {"go2rtc_url": str(data.get("go2rtc_url") or "http://localhost:1984").rstrip("/")}
    domain = str(data.get("domain") or "").strip().lower()
    if domain:
        if not re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", domain):
            raise ValueError("Domain name is not valid (e.g. camera.example.com)")
        cfg["auth"] = {"enabled": True, "rp_id": domain, "origin": f"https://{domain}"}
        cfg["api"]["public_url"] = f"https://{domain}"
    return cfg


def go2rtc_yaml(cfg: dict) -> str:
    streams = {c["name"]: [c["main_url"]] for c in cfg["cameras"]}
    return yaml.safe_dump({"streams": streams, "api": {"listen": ":1984"}}, sort_keys=False)


def probe_rtsp(url: str, timeout: float = 8.0) -> dict:
    """Fetch one frame: does the address work, and at what resolution? With a small JPEG preview."""
    import av
    import cv2
    try:
        with av.open(url, options={"rtsp_transport": "tcp", "stimeout": str(int(timeout * 1e6))},
                     timeout=(timeout, timeout)) as c:
            st = c.streams.video[0]
            for frame in c.decode(st):
                img = frame.to_ndarray(format="bgr24")
                h, w = img.shape[:2]
                small = cv2.resize(img, (320, int(320 * h / w)))
                ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
                import base64
                return {"ok": True, "width": w, "height": h, "codec": st.codec_context.name,
                        "preview": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()}
    except Exception as exc:  # noqa: BLE001 - error message for the user
        msg = str(exc)
        if "401" in msg or "Unauthorized" in msg:
            msg = "User name or password is wrong"
        elif "timed out" in msg.lower() or "timeout" in msg.lower():
            msg = "No answer: are the IP address and port correct?"
        return {"ok": False, "error": msg[:200]}
    return {"ok": False, "error": "No image received"}


def test_mqtt(host: str, port: int, username: str | None, password: str | None) -> dict:
    import paho.mqtt.client as mqtt
    done = threading.Event()
    result: dict = {"ok": False, "error": "No answer from the MQTT server"}

    def on_connect(_c, _u, _f, rc, _p):
        result.update({"ok": not rc.is_failure, "error": None if not rc.is_failure else str(rc)})
        done.set()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"camsvc-setup-{secrets.token_hex(3)}")
    if username:
        client.username_pw_set(username, password)
    client.on_connect = on_connect
    try:
        client.connect_async(host, port, keepalive=10)
        client.loop_start()
        done.wait(6)
    except OSError as exc:
        result["error"] = str(exc)
    finally:
        client.loop_stop()
        client.disconnect()
    return result


def push_go2rtc(url: str, cfg: dict) -> None:
    """Pass the streams to a running go2rtc right away (otherwise after the next restart)."""
    for cam in cfg["cameras"]:
        q = urllib.parse.urlencode({"name": cam["name"], "src": cam["main_url"]})
        try:
            urllib.request.urlopen(urllib.request.Request(f"{url}/api/streams?{q}", method="PUT"), timeout=3)
        except OSError:
            pass


def setup_token(state_dir: Path) -> str:
    state_dir.mkdir(parents=True, exist_ok=True)
    f = state_dir / TOKEN_FILE
    if f.exists():
        return f.read_text().strip()
    code = "-".join(secrets.token_hex(2).upper() for _ in range(2))
    f.write_text(code + "\n")
    f.chmod(0o600)
    return code


def create_setup_app(config_path: Path, state_dir: Path, reason: str):
    token = setup_token(state_dir)
    log.warning("Setup wizard active (%s). Open the web page and use setup code: %s", reason, token)
    app = FastAPI(title="Uitkijk setup")

    def check(request: Request) -> None:
        given = request.headers.get("x-setup-code", "")
        if not hmac.compare_digest(given.strip().upper().encode(), token.encode()):
            raise HTTPException(401, "Setup code is not correct (see the container log)")

    @app.get("/", response_class=HTMLResponse)
    def page():
        return HTMLResponse((WEB_DIR / "setup.html").read_text(encoding="utf-8"))

    @app.get("/login")
    def login():
        return RedirectResponse("/")

    @app.get("/icon-{size}.png")
    def icon(size: int):
        f = WEB_DIR / f"icon-{size}.png"
        if size not in (180, 512) or not f.exists():
            raise HTTPException(404)
        return FileResponse(f, media_type="image/png")

    @app.get("/api/setup/info")
    def info(request: Request):
        host = (request.headers.get("host") or "localhost:8080").split(",")[0]
        return {"reason": reason, "brands": {k: {"name": v["name"], "port": v["port"],
                                                  "custom": not v["main"]} for k, v in BRANDS.items()},
                "public_url": f"http://{host}"}

    @app.post("/api/setup/check")
    def check_code(request: Request):
        check(request)
        return {"ok": True}

    @app.post("/api/setup/test-camera")
    def test_camera(request: Request, cam: dict = Body(...)):
        check(request)
        try:
            main, sub = camera_urls(cam)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        res = probe_rtsp(main)
        if res["ok"] and sub:
            res["sub"] = {k: v for k, v in probe_rtsp(sub).items() if k != "preview"}
        return res

    @app.post("/api/setup/test-mqtt")
    def mqtt_check(request: Request, m: dict = Body(...)):
        check(request)
        if not m.get("host"):
            raise HTTPException(400, "Enter the MQTT server")
        return test_mqtt(str(m["host"]), int(m.get("port") or 1883), m.get("username") or None,
                         m.get("password") or None)

    @app.post("/api/setup/save")
    def save(request: Request, data: dict = Body(...)):
        check(request)
        # paths and port follow the environment the wizard runs in (Docker: /data and /clips)
        data["port"] = int(os.environ.get("CAMSVC_PORT", "8080"))
        data["state_dir"] = str(state_dir)
        data["clips_dir"] = os.environ.get("CAMSVC_CLIPS", "/clips")
        data["model_path"] = os.environ.get("CAMSVC_MODEL", "/data/models/yolo11n.onnx")
        try:
            cfg = build_config(data)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        config_path.parent.mkdir(parents=True, exist_ok=True)
        if config_path.exists():
            config_path.rename(config_path.with_suffix(".yaml.previous"))
        header = ("# Created by the setup wizard. Feel free to edit; restart the service afterwards.\n"
                  "# All options are listed in config.example.yaml.\n")
        config_path.write_text(header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        config_path.chmod(0o600)
        if cfg.get("live"):
            g = config_path.parent / "go2rtc" / "go2rtc.yaml"
            g.parent.mkdir(parents=True, exist_ok=True)
            g.write_text(go2rtc_yaml(cfg))
            g.chmod(0o600)            # contains the camera passwords
            push_go2rtc(cfg["live"]["go2rtc_url"], cfg)
        (state_dir / TOKEN_FILE).unlink(missing_ok=True)
        log.warning("Config saved to %s; restarting in normal mode", config_path)
        # stop a moment later so the response still arrives; Docker/systemd restarts us
        threading.Timer(1.5, lambda: os._exit(3)).start()
        return JSONResponse({"ok": True, "public_url": cfg["api"]["public_url"],
                             "auth": bool(cfg.get("auth"))})

    return app
