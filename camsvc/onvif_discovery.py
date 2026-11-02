"""RTSP-URL's ophalen via ONVIF als die niet in de config staan."""

from __future__ import annotations

import logging
from urllib.parse import quote, urlsplit, urlunsplit

log = logging.getLogger(__name__)


def _with_credentials(uri: str, user: str, password: str) -> str:
    parts = urlsplit(uri)
    if parts.username:
        return uri
    netloc = f"{quote(user, safe='')}:{quote(password, safe='')}@{parts.hostname}"
    if parts.port:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def discover_streams(host: str, user: str, password: str, port: int = 80) -> tuple[str, str]:
    """Geeft (main_url, sub_url): het profiel met de hoogste en de laagste resolutie."""
    from onvif import ONVIFCamera  # pas importeren als het nodig is

    cam = ONVIFCamera(host, port, user, password)
    media = cam.create_media_service()
    found: list[tuple[int, str]] = []
    for profile in media.GetProfiles():
        venc = getattr(profile, "VideoEncoderConfiguration", None)
        if venc is None or venc.Resolution is None:
            continue
        req = media.create_type("GetStreamUri")
        req.ProfileToken = profile.token
        req.StreamSetup = {"Stream": "RTP-Unicast", "Transport": {"Protocol": "RTSP"}}
        uri = media.GetStreamUri(req).Uri
        pixels = int(venc.Resolution.Width) * int(venc.Resolution.Height)
        found.append((pixels, _with_credentials(uri, user, password)))
        log.info("ONVIF %s: profiel %s %sx%s", host, profile.token,
                 venc.Resolution.Width, venc.Resolution.Height)
    if not found:
        raise RuntimeError(f"ONVIF {host}: geen videoprofielen gevonden")
    found.sort()
    return found[-1][1], found[0][1]
