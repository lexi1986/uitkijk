"""REST-API (alleen voor het LAN): status, clips, snapshots en maskercontrole."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import asyncio
import logging
import urllib.error
import urllib.parse
import urllib.request

from fastapi import Body, FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response

from . import __version__
from .config import EDITABLE, validate_settings
from .motion import draw_mask_overlay, validate_mask, validate_zones
from .summary import compare_ai, parse_day, summarize_day

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
# speler-bestanden van go2rtc die via /live/ doorgegeven worden
LIVE_FILES = {"stream.html": "text/html", "video-stream.js": "text/javascript",
              "video-rtc.js": "text/javascript"}
LABELS = ("aanbellen", "persoon", "voertuig", "dier", "beweging")

if TYPE_CHECKING:
    from .service import Service


PUBLIC_PATHS = ("/login", "/manifest.json", "/icon-180.png", "/icon-512.png")


def create_app(svc: "Service") -> FastAPI:
    app = FastAPI(title="camera-service", version=__version__, docs_url="/docs", redoc_url=None)
    clips_root = svc.cfg.clips_dir.resolve()

    def get_cam(name: str):
        cam = svc.cameras.get(name)
        if cam is None:
            raise HTTPException(404, f"Onbekende camera '{name}'")
        return cam

    # --- inloggen met passkeys ---------------------------------------------
    from .auth import COOKIE, AuthError

    def direct_local(client_host: str | None, headers) -> bool:
        """Rechtstreeks vanaf de server zelf (niet via Caddy, die zet altijd X-Forwarded-For)."""
        return client_host in ("127.0.0.1", "::1") and "x-forwarded-for" not in headers

    def logged_in(request: Request) -> bool:
        return (svc.auth is None or direct_local(request.client.host if request.client else None,
                                                 request.headers)
                or svc.auth.session_ok(request.cookies.get(COOKIE)))

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        path = request.url.path
        if path in PUBLIC_PATHS or path.startswith("/api/auth/") or logged_in(request):
            return await call_next(request)
        if request.method == "GET" and "text/html" in request.headers.get("accept", ""):
            target = path + (f"?{request.url.query}" if request.url.query else "")
            return RedirectResponse("/login" + (f"?next={urllib.parse.quote(target)}" if target != "/" else ""))
        return JSONResponse({"detail": "Niet ingelogd"}, status_code=401)

    def auth_or_404():
        if svc.auth is None:
            raise HTTPException(404, "Inloggen staat uit (auth.enabled in de config)")
        return svc.auth

    def with_cookie(body: dict, cookie: str | None) -> JSONResponse:
        resp = JSONResponse(body)
        if cookie:
            resp.set_cookie(COOKIE, cookie, max_age=svc.auth.session_seconds, httponly=True,
                            secure=True, samesite="lax", path="/")
        return resp

    @app.get("/login", response_class=HTMLResponse, include_in_schema=False)
    def login_page():
        return HTMLResponse((WEB_DIR / "login.html").read_text(encoding="utf-8"))

    @app.get("/api/auth/state")
    def auth_state(request: Request):
        if svc.auth is None:
            return {"enabled": False, "logged_in": True, "setup": False}
        return {"enabled": True, "logged_in": logged_in(request), "setup": not svc.auth.credentials()}

    @app.post("/api/auth/register/options")
    def auth_register_options(request: Request, payload: dict = Body(default={})):
        try:
            return auth_or_404().registration_options(logged_in(request), payload.get("code"))
        except AuthError as exc:
            raise HTTPException(403, str(exc)) from exc

    @app.post("/api/auth/register")
    def auth_register(request: Request, payload: dict = Body(...)):
        auth = auth_or_404()
        was_in = logged_in(request)
        try:
            cred_id = auth.register(str(payload.get("challenge_id", "")), payload.get("credential") or {},
                                    str(payload.get("name", "")), was_in, payload.get("code"))
        except AuthError as exc:
            raise HTTPException(403, str(exc)) from exc
        return with_cookie({"ok": True}, None if was_in else auth.make_cookie(cred_id))

    @app.post("/api/auth/login/options")
    def auth_login_options():
        return auth_or_404().login_options()

    @app.post("/api/auth/login")
    def auth_login(payload: dict = Body(...)):
        auth = auth_or_404()
        try:
            cookie = auth.login(str(payload.get("challenge_id", "")), payload.get("credential") or {})
        except AuthError as exc:
            raise HTTPException(403, str(exc)) from exc
        return with_cookie({"ok": True}, cookie)

    @app.post("/api/auth/logout")
    def auth_logout():
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(COOKIE, path="/")
        return resp

    @app.get("/api/auth/credentials")
    def auth_credentials(request: Request):
        if not logged_in(request):
            raise HTTPException(401, "Niet ingelogd")
        return {"credentials": auth_or_404().credentials()}

    @app.delete("/api/auth/credentials/{cred_id}")
    def auth_remove(cred_id: str, request: Request):
        if not logged_in(request):
            raise HTTPException(401, "Niet ingelogd")
        if not auth_or_404().remove(cred_id):
            raise HTTPException(404, "Niet gevonden")
        return {"credentials": svc.auth.credentials()}

    @app.post("/api/auth/logout-all")
    def auth_logout_all(request: Request):
        if not logged_in(request):
            raise HTTPException(401, "Niet ingelogd")
        auth_or_404().logout_everywhere()
        return {"ok": True}

    @app.get("/manifest.json", include_in_schema=False)
    def manifest():
        """Voor 'Zet op beginscherm': eigen icoon, donkere achtergrond, zonder browserbalk."""
        return JSONResponse({
            "name": "Camera's", "short_name": "Camera's", "start_url": "/", "display": "standalone",
            "background_color": "#090e18", "theme_color": "#090e18", "lang": "nl",
            "icons": [{"src": "/icon-180.png", "sizes": "180x180", "type": "image/png"},
                      {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"}]})

    @app.get("/icon-{size}.png", include_in_schema=False)
    def icon(size: int):
        f = WEB_DIR / f"icon-{size}.png"
        if size not in (180, 512) or not f.exists():
            raise HTTPException(404, "Niet gevonden")
        return FileResponse(f, media_type="image/png", headers={"Cache-Control": "max-age=86400"})

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/cameras")
    def cameras():
        """Camera's met status, voor de webpagina."""
        return [{"name": c.name, "friendly_name": c.cfg.friendly_name, "online": c.online,
                 "motion": c.motion_active, "event_active": c.event is not None,
                 "detect": c.detect_enabled, "record": c.record_enabled,
                 "loitering": c.loitering, "face": c.face_name if c.faces else None,
                 "live": f"/live/stream.html?src={urllib.parse.quote(c.cfg.live_stream)}"
                         f"&mode={urllib.parse.quote(c.cfg.live_mode, safe=',')}"
                 if svc.cfg.go2rtc_url and c.cfg.live_stream else None}
                for c in svc.cameras.values()]

    @app.get("/api/events")
    def all_events(camera: str | None = None, label: str | None = None,
                   offset: int = 0, limit: int = 60, person: str | None = None,
                   plate: str | None = None, q: str | None = None, day: str | None = None,
                   belangrijk: bool = False):
        """Alle opgeslagen events (nieuwste eerst), te filteren op camera, label, persoon,
        kenteken en vrije tekst `q` (namen, kentekens, AI-beschrijving; alle woorden moeten passen)."""
        if camera:
            get_cam(camera)
        if label and label not in LABELS:
            raise HTTPException(400, f"Onbekend label '{label}'")
        if day:
            try:
                day = parse_day(day).isoformat()
            except ValueError as exc:
                raise HTTPException(400, "Datum als JJJJ-MM-DD") from exc
        items, total = svc.store.browse(camera, label, max(offset, 0), min(max(limit, 1), 200),
                                        person or None, plate or None, (q or "").strip()[:100] or None,
                                        svc.plates.known if svc.plates else {}, day or None,
                                        belangrijk)
        return {"total": total, "offset": offset, "items": items}

    @app.get("/api/summary")
    def summary(day: str | None = None):
        """Dagoverzicht (standaard vandaag): tellingen, personen, auto's, pakketjes en een tekst."""
        try:
            d = parse_day(day)
        except ValueError as exc:
            raise HTTPException(400, "Datum als JJJJ-MM-DD") from exc
        return summarize_day(svc.cfg.clips_dir, d, svc.plates.known if svc.plates else {})

    @app.post("/api/events/vals")
    def event_false(payload: dict = Body(...)):
        """Event als vals alarm markeren: {"clip": "<camera>/<datum>/<tijd>_<label>.mp4", "vals": true}.
        De camera onthoudt dan die plek voor dat soort object; "vals": false draait het terug."""
        try:
            return svc.mark_false(str(payload.get("clip", "")), bool(payload.get("vals", True)))
        except KeyError as exc:
            raise HTTPException(404, "Event niet gevonden") from exc

    @app.get("/api/cameras/{name}/false-spots")
    def false_spots(name: str):
        """Plekken die als vals alarm gemarkeerd zijn (box als fractie van het beeld)."""
        return {"spots": get_cam(name).false_spots}

    @app.delete("/api/cameras/{name}/false-spots")
    def false_spots_clear(name: str):
        return {"verwijderd": get_cam(name).remove_false_spot()}

    @app.get("/api/ai-vergelijking")
    def ai_compare(days: int = 7):
        """Externe AI (HA) naast de lokale AI, over de laatste `days` dagen."""
        return compare_ai(svc.cfg.clips_dir, parse_day(None), min(max(days, 1), 31))

    @app.get("/api/cameras/{name}/frame.jpg")
    def frame(name: str):
        """Recentste beeld van de sub-stream (ververst minstens elke paar seconden)."""
        cam = get_cam(name)
        if cam.latest_bgr is None:
            raise HTTPException(503, "Nog geen beeld")
        ok, buf = cv2.imencode(".jpg", cam.latest_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # --- kentekens ---------------------------------------------------------
    @app.get("/api/plates")
    def plates_list():
        """Bekende kentekens met hun naam."""
        if svc.plates is None:
            return {"enabled": False, "known": {}}
        return {"enabled": True, "known": dict(sorted(svc.plates.known.items()))}

    @app.put("/api/plates/{plate}")
    def plate_name(plate: str, payload: dict = Body(...)):
        """Naam geven aan een kenteken: {"name": "Onze auto"}."""
        if svc.plates is None:
            raise HTTPException(404, "Kentekenherkenning staat uit")
        try:
            svc.plates.set_name(plate, str(payload.get("name", "")))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"known": svc.plates.known}

    @app.delete("/api/plates/{plate}")
    def plate_forget(plate: str):
        if svc.plates is None or not svc.plates.forget(plate):
            raise HTTPException(404, "Niet gevonden")
        return {"known": svc.plates.known}

    # --- gezichten ---------------------------------------------------------
    def faces_or_404():
        if svc.faces is None:
            raise HTTPException(404, "Gezichtsherkenning staat uit (faces.enabled in de config)")
        return svc.faces

    def clip_file(rel: str) -> Path:
        target = (clips_root / rel).resolve()
        if clips_root not in target.parents or not target.is_file():
            raise HTTPException(404, "Niet gevonden")
        return target

    def faces_payload(faces) -> dict:
        return {"enabled": True, "model": faces.backend.name, "people": faces.people(),
                "auto": faces.auto_samples(), "unknown": faces.unknown.count()}

    @app.get("/api/faces")
    def faces_list():
        """Bekende personen met hun voorbeeldfoto's (zelf en automatisch geleerd)."""
        if svc.faces is None:
            return {"enabled": False, "people": {}, "auto": {}, "unknown": 0}
        return faces_payload(svc.faces)

    @app.get("/api/faces/{name}/{sample_id}.jpg")
    def face_sample(name: str, sample_id: str):
        path = faces_or_404().sample_path(name, sample_id)
        if path is None:
            raise HTTPException(404, "Niet gevonden")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/faces/{name}/auto/{sample_id}.jpg")
    def face_auto_sample(name: str, sample_id: str):
        path = faces_or_404().sample_path(name, sample_id, auto=True)
        if path is None:
            raise HTTPException(404, "Niet gevonden")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/faces/{name}")
    def face_learn(name: str, payload: dict = Body(...)):
        """Leer een gezicht uit een event: {"crop": "<camera>/<datum>/<tijd>_<label>.face1.jpg"}."""
        faces = faces_or_404()
        crop = str(payload.get("crop", ""))
        if ".face" not in crop or not crop.endswith(".jpg"):
            raise HTTPException(400, "Geen gezichtsuitsnede")
        target = clip_file(crop)
        try:
            sample_id = faces.add(name, target.read_bytes())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        # het event zelf ook bijwerken, zodat de tijdlijn de naam toont
        svc.store.set_face_name(crop, name)
        faces.unknown.drop_crop(crop)
        return {"name": name, "sample": sample_id, **faces_payload(faces)}

    @app.delete("/api/faces/{name}/auto/{sample_id}")
    def face_forget_auto(name: str, sample_id: str):
        if not faces_or_404().remove(name, sample_id, auto=True):
            raise HTTPException(404, "Niet gevonden")
        return faces_payload(svc.faces)

    @app.delete("/api/faces/{name}/{sample_id}")
    def face_forget_sample(name: str, sample_id: str):
        if not faces_or_404().remove(name, sample_id):
            raise HTTPException(404, "Niet gevonden")
        return faces_payload(svc.faces)

    @app.delete("/api/faces/{name}")
    def face_forget(name: str):
        if not faces_or_404().remove(name):
            raise HTTPException(404, "Niet gevonden")
        return faces_payload(svc.faces)

    # --- onbekende gezichten (los pad: een persoon mag ook 'unknown' heten) ---
    def ids_from(payload: dict) -> list[str]:
        ids = payload.get("ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
            raise HTTPException(400, "ids ontbreekt")
        return ids[:500]

    @app.get("/api/unknown-faces")
    def unknown_groups():
        """Onbekende gezichten, gegroepeerd per (waarschijnlijk) dezelfde persoon."""
        return {"groups": faces_or_404().unknown.groups()}

    @app.get("/api/unknown-faces/{sample_id}.jpg")
    def unknown_sample(sample_id: str):
        path = faces_or_404().unknown.path(sample_id)
        if path is None:
            raise HTTPException(404, "Niet gevonden")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/unknown-faces/assign")
    def unknown_assign(payload: dict = Body(...)):
        """{"ids": [...], "name": "Sam"}: groep leren en alle events met die gezichten bijwerken."""
        faces = faces_or_404()
        try:
            updates = faces.unknown.assign(ids_from(payload), str(payload.get("name", "")).strip())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, "Niet gevonden") from exc
        events = sum(svc.store.set_face_name(crop, name) for crop, name in updates)
        return {"faces": len(updates), "events": events, **faces_payload(faces),
                "groups": faces.unknown.groups()}

    @app.post("/api/unknown-faces/dismiss")
    def unknown_dismiss(payload: dict = Body(...)):
        faces = faces_or_404()
        faces.unknown.dismiss(ids_from(payload))
        return {**faces_payload(faces), "groups": faces.unknown.groups()}

    # --- live beeld via go2rtc ---------------------------------------------
    # Browsers blokkeren steeds vaker verzoeken van een http-pagina naar een ander adres in
    # het eigen netwerk (Local Network Access). Daarom loopt go2rtc via dit adres: alleen de
    # spelerbestanden en de streams van de eigen camera's.
    def live_streams() -> set[str]:
        return {c.cfg.live_stream for c in svc.cameras.values() if c.cfg.live_stream}

    @app.get("/live/{name}", include_in_schema=False)
    def live_file(name: str):
        if not svc.cfg.go2rtc_url or name not in LIVE_FILES:
            raise HTTPException(404, "Niet gevonden")
        try:
            with urllib.request.urlopen(f"{svc.cfg.go2rtc_url}/{name}", timeout=5) as r:
                body = r.read()
        except (urllib.error.URLError, OSError) as exc:
            raise HTTPException(502, f"go2rtc niet bereikbaar: {exc}") from exc
        return Response(body, media_type=LIVE_FILES[name])

    @app.websocket("/live/api/ws")
    async def live_ws(ws: WebSocket):
        import websockets

        src = ws.query_params.get("src", "")
        if svc.auth is not None and not (
                direct_local(ws.client.host if ws.client else None, ws.headers)
                or svc.auth.session_ok(ws.cookies.get(COOKIE))):
            await ws.close(code=1008)
            return
        if not svc.cfg.go2rtc_url or src not in live_streams():
            await ws.close(code=1008)
            return
        await ws.accept()
        url = svc.cfg.go2rtc_url.replace("http", "ws", 1) + "/api/ws?src=" + urllib.parse.quote(src)
        try:
            async with websockets.connect(url, max_size=None, open_timeout=5) as up:
                async def to_browser():
                    async for msg in up:
                        if isinstance(msg, bytes):
                            await ws.send_bytes(msg)
                        else:
                            await ws.send_text(msg)

                async def to_go2rtc():
                    while True:
                        msg = await ws.receive()
                        if msg["type"] == "websocket.disconnect":
                            return
                        if msg.get("text") is not None:
                            await up.send(msg["text"])
                        elif msg.get("bytes") is not None:
                            await up.send(msg["bytes"])

                tasks = [asyncio.create_task(to_browser()), asyncio.create_task(to_go2rtc())]
                _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
        except Exception as exc:  # noqa: BLE001 - verbinding weg of go2rtc onbereikbaar
            log.debug("live %s gestopt: %s", src, exc)
        finally:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001 - browser was al weg (venster gesloten)
                pass

    @app.get("/api/status")
    def status():
        return {
            "version": __version__,
            "mqtt_connected": svc.mqtt.connected.is_set() if svc.mqtt else None,   # None = geen MQTT
            "storage": {"clips_dir": str(clips_root), "used_gb": round(svc.retention.used_gb, 2),
                        "max_gb": svc.cfg.retention_max_gb, "retention_days": svc.cfg.retention_days},
            "detector": getattr(svc.detector, "device", None),
            "ai": svc.ai.status() if svc.ai else {"state": "uit"},
            "health": svc.health_snapshot(),
            "cameras": [c.status() for c in svc.cameras.values()],
        }

    @app.get("/api/cameras/{name}/events")
    def events(name: str):
        get_cam(name)
        return svc.store.as_payload(name)

    @app.get("/api/cameras/{name}/latest.jpg")
    def latest(name: str):
        cam = get_cam(name)
        if cam.latest_snapshot_jpg:
            return Response(cam.latest_snapshot_jpg, media_type="image/jpeg")
        if cam.latest_bgr is not None:
            ok, buf = cv2.imencode(".jpg", cam.latest_bgr)
            if ok:
                return Response(buf.tobytes(), media_type="image/jpeg")
        raise HTTPException(503, "Nog geen beeld")

    @app.post("/api/cameras/{name}/trigger")
    def trigger(name: str, label: str = "aanbellen"):
        """Event starten van buitenaf, bijv. bij de deurbelknop."""
        if label not in ("aanbellen", "beweging"):
            raise HTTPException(400, "label moet aanbellen of beweging zijn")
        if not get_cam(name).manual_trigger(label):
            raise HTTPException(503, "Camera offline")
        return {"ok": True}

    @app.get("/api/cameras/{name}/settings")
    def settings_get(name: str):
        """Aanpasbare instellingen met hun grenzen."""
        get_cam(name)
        schema = {p: {"type": k.__name__, "min": lo, "max": hi, "label": label}
                  for p, (k, lo, hi, label) in EDITABLE.items()}
        return {"values": svc.camera_settings(name), "schema": schema}

    @app.put("/api/cameras/{name}/settings")
    def settings_set(name: str, payload: dict = Body(...)):
        """Instellingen wijzigen; direct actief en bewaard over herstarts."""
        get_cam(name)
        try:
            values = validate_settings(payload)
            return {"values": svc.set_settings(name, values)}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/cameras/{name}/mask")
    def mask_get(name: str):
        """Huidig masker: lijst vlakken, punten [x, y] in procenten."""
        return {"mask": get_cam(name).cfg.mask}

    @app.put("/api/cameras/{name}/mask")
    def mask_set(name: str, payload: dict = Body(...)):
        """Masker instellen vanuit de webpagina; direct actief en bewaard over herstarts."""
        get_cam(name)
        try:
            polygons = validate_mask(payload.get("mask"))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        svc.set_mask(name, polygons)
        return {"mask": polygons}

    @app.get("/api/cameras/{name}/zones")
    def zones_get(name: str):
        """Zones: objecten tellen alleen mee als ze met hun onderkant in een zone staan."""
        return {"zones": get_cam(name).cfg.zones}

    @app.put("/api/cameras/{name}/zones")
    def zones_set(name: str, payload: dict = Body(...)):
        get_cam(name)
        try:
            zones = validate_zones(payload.get("zones"))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        svc.set_zones(name, zones)
        return {"zones": zones}

    @app.get("/api/cameras/{name}/mask.jpg")
    def mask(name: str):
        """Laatste frame van de sub-stream met het masker en een procentenraster erover."""
        cam = get_cam(name)
        if cam.latest_bgr is None:
            raise HTTPException(503, "Nog geen beeld; wacht tot de sub-stream draait")
        img = draw_mask_overlay(cam.latest_bgr, cam.cfg.mask, cam.cfg.zones)
        ok, buf = cv2.imencode(".jpg", img)
        return Response(buf.tobytes(), media_type="image/jpeg")

    @app.get("/clips/{path:path}")
    def clip(path: str):
        target = (clips_root / path).resolve()
        if clips_root not in target.parents or not target.is_file():
            raise HTTPException(404, "Niet gevonden")
        if target.suffix not in (".mp4", ".jpg", ".json"):
            raise HTTPException(404, "Niet gevonden")
        media = {".mp4": "video/mp4", ".jpg": "image/jpeg", ".json": "application/json"}[target.suffix]
        return FileResponse(target, media_type=media)

    return app


__all__ = ["create_app", "Path"]
