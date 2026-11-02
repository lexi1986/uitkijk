"""Fetch RTSP URLs via ONVIF when they are not in the config."""

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
    """Returns (main_url, sub_url): the profile with the highest and the lowest resolution."""
    from onvif import ONVIFCamera  # only import when needed

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
        log.info("ONVIF %s: profile %s %sx%s", host, profile.token,
                 venc.Resolution.Width, venc.Resolution.Height)
    if not found:
        raise RuntimeError(f"ONVIF {host}: no video profiles found")
    found.sort()
    return found[-1][1], found[0][1]
