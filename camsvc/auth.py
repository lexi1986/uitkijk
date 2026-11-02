"""Inloggen met passkeys (WebAuthn), zonder wachtwoorden.

- Passkeys werken alleen via HTTPS met een domeinnaam (bijv. https://camera.voorbeeld.nl
  achter Caddy). rp_id is die domeinnaam, origin de volledige https-URL.
- De eerste passkey koppel je met een eenmalige koppelcode (staat in het log en in
  <state_dir>/koppelcode). Daarna voeg je ingelogd extra apparaten toe; met iCloud-sleutelhanger
  staat een passkey van de MacBook ook meteen op de iPhone.
- Na inloggen een ondertekende cookie (HttpOnly, Secure) die `session_days` geldig blijft.
  "Overal uitloggen" vervangt het geheim, waarmee alle cookies vervallen.
- Verzoeken direct vanaf de server zelf (127.0.0.1, niet via de proxy) hoeven niet in te loggen:
  handig voor controles en scripts op de server. Caddy zet altijd X-Forwarded-For en telt dus niet.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
from pathlib import Path

from webauthn import (generate_authentication_options, generate_registration_options,
                      options_to_json, verify_authentication_response,
                      verify_registration_response)
from webauthn.helpers import base64url_to_bytes, bytes_to_base64url
from webauthn.helpers.structs import (AuthenticatorSelectionCriteria, PublicKeyCredentialDescriptor,
                                      ResidentKeyRequirement, UserVerificationRequirement)

log = logging.getLogger(__name__)

COOKIE = "camsvc_sessie"
CHALLENGE_TTL = 300.0
USER_ID = b"camera-service-huishouden"   # één gedeeld account: elk apparaat is een eigen passkey


class AuthError(Exception):
    pass


class Auth:
    def __init__(self, state_dir: Path, rp_id: str, origin: str, session_days: float = 90):
        self.rp_id = rp_id
        self.origin = origin
        self.session_seconds = int(session_days * 86400)
        self._file = state_dir / "auth.json"
        self._code_file = state_dir / "koppelcode"
        self._lock = threading.Lock()
        self._challenges: dict[str, tuple[bytes, float, str]] = {}   # id -> (challenge, tot, soort)
        data = self._load()
        if not data.get("secret"):
            data["secret"] = secrets.token_hex(32)
            self._save(data)
        if not data.get("credentials") and not self._code_file.exists():
            self.new_setup_code()       # bestaande code blijft geldig over herstarts heen

    # ------------------------------------------------------------------ opslag
    def _load(self) -> dict:
        try:
            return json.loads(self._file.read_text())
        except (FileNotFoundError, ValueError):
            return {}

    def _save(self, data: dict) -> None:
        tmp = self._file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.chmod(0o600)
        tmp.replace(self._file)

    def credentials(self) -> list[dict]:
        return [{k: c[k] for k in ("id", "naam", "aangemaakt", "laatst")}
                for c in self._load().get("credentials", [])]

    def new_setup_code(self) -> str:
        code = "-".join(secrets.token_hex(2).upper() for _ in range(3))
        self._code_file.write_text(code + "\n")
        self._code_file.chmod(0o600)
        log.warning("Passkey-koppelcode: %s (eenmalig, ook in %s)", code, self._code_file)
        return code

    def _setup_code_ok(self, code: str | None) -> bool:
        try:
            expected = self._code_file.read_text().strip()
        except FileNotFoundError:
            return False
        return bool(code) and hmac.compare_digest(expected.encode(), code.strip().upper().encode())

    # ------------------------------------------------------------------ sessies
    def _sign(self, payload: bytes, secret: str) -> str:
        return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

    def make_cookie(self, cred_id: str) -> str:
        payload = base64.urlsafe_b64encode(json.dumps(
            {"c": cred_id, "e": int(time.time()) + self.session_seconds}).encode()).decode()
        return f"{payload}.{self._sign(payload.encode(), self._load()['secret'])}"

    def session_ok(self, cookie: str | None) -> bool:
        if not cookie or "." not in cookie:
            return False
        payload, sig = cookie.rsplit(".", 1)
        data = self._load()
        if not hmac.compare_digest(sig, self._sign(payload.encode(), data.get("secret", ""))):
            return False
        try:
            info = json.loads(base64.urlsafe_b64decode(payload.encode()))
        except ValueError:
            return False
        # een verwijderde passkey logt ook dat apparaat uit
        return info.get("e", 0) > time.time() and any(
            c["id"] == info.get("c") for c in data.get("credentials", []))

    def logout_everywhere(self) -> None:
        with self._lock:
            data = self._load()
            data["secret"] = secrets.token_hex(32)
            self._save(data)

    # ------------------------------------------------------------------ challenges
    def _new_challenge(self, challenge: bytes, kind: str) -> str:
        now = time.monotonic()
        self._challenges = {k: v for k, v in self._challenges.items() if v[1] > now}
        cid = secrets.token_urlsafe(16)
        self._challenges[cid] = (challenge, now + CHALLENGE_TTL, kind)
        return cid

    def _take_challenge(self, cid: str, kind: str) -> bytes:
        item = self._challenges.pop(cid or "", None)
        if item is None or item[1] < time.monotonic() or item[2] != kind:
            raise AuthError("Verlopen, probeer het opnieuw")
        return item[0]

    # ------------------------------------------------------------------ registreren
    def registration_options(self, logged_in: bool, setup_code: str | None) -> dict:
        if not logged_in and not self._setup_code_ok(setup_code):
            raise AuthError("Koppelcode klopt niet")
        existing = self._load().get("credentials", [])
        opts = generate_registration_options(
            rp_id=self.rp_id, rp_name="Camera's", user_id=USER_ID, user_name="huishouden",
            user_display_name="Camera's thuis",
            authenticator_selection=AuthenticatorSelectionCriteria(
                resident_key=ResidentKeyRequirement.REQUIRED,
                user_verification=UserVerificationRequirement.REQUIRED),
            exclude_credentials=[PublicKeyCredentialDescriptor(id=base64url_to_bytes(c["id"]))
                                 for c in existing])
        return {"challenge_id": self._new_challenge(opts.challenge, "reg"),
                "options": json.loads(options_to_json(opts))}

    def register(self, challenge_id: str, credential: dict, name: str, logged_in: bool,
                 setup_code: str | None) -> str:
        if not logged_in and not self._setup_code_ok(setup_code):
            raise AuthError("Koppelcode klopt niet")
        challenge = self._take_challenge(challenge_id, "reg")
        try:
            v = verify_registration_response(credential=credential, expected_challenge=challenge,
                                             expected_rp_id=self.rp_id, expected_origin=self.origin,
                                             require_user_verification=True)
        except Exception as exc:  # noqa: BLE001 - de bibliotheek gebruikt diverse fouttypes
            raise AuthError(f"Passkey niet geaccepteerd: {exc}") from exc
        cred_id = bytes_to_base64url(v.credential_id)
        today = time.strftime("%Y-%m-%d")
        with self._lock:
            data = self._load()
            data.setdefault("credentials", []).append({
                "id": cred_id, "public_key": bytes_to_base64url(v.credential_public_key),
                "sign_count": v.sign_count, "naam": (name or "Apparaat").strip()[:40],
                "aangemaakt": today, "laatst": today})
            self._save(data)
        self._code_file.unlink(missing_ok=True)       # koppelcode is eenmalig
        log.info("Passkey toegevoegd: %s", name)
        return cred_id

    def remove(self, cred_id: str) -> bool:
        with self._lock:
            data = self._load()
            before = len(data.get("credentials", []))
            data["credentials"] = [c for c in data.get("credentials", []) if c["id"] != cred_id]
            if len(data["credentials"]) == before:
                return False
            self._save(data)
        if not data["credentials"]:
            self.new_setup_code()
        return True

    # ------------------------------------------------------------------ inloggen
    def login_options(self) -> dict:
        opts = generate_authentication_options(
            rp_id=self.rp_id, user_verification=UserVerificationRequirement.REQUIRED)
        return {"challenge_id": self._new_challenge(opts.challenge, "auth"),
                "options": json.loads(options_to_json(opts))}

    def login(self, challenge_id: str, credential: dict) -> str:
        challenge = self._take_challenge(challenge_id, "auth")
        cred_id = credential.get("id", "")
        with self._lock:
            data = self._load()
            stored = next((c for c in data.get("credentials", []) if c["id"] == cred_id), None)
            if stored is None:
                raise AuthError("Onbekende passkey")
            try:
                v = verify_authentication_response(
                    credential=credential, expected_challenge=challenge, expected_rp_id=self.rp_id,
                    expected_origin=self.origin,
                    credential_public_key=base64url_to_bytes(stored["public_key"]),
                    credential_current_sign_count=stored["sign_count"], require_user_verification=True)
            except Exception as exc:  # noqa: BLE001
                raise AuthError("Inloggen mislukt") from exc
            stored["sign_count"] = v.new_sign_count
            stored["laatst"] = time.strftime("%Y-%m-%d")
            self._save(data)
        return self.make_cookie(cred_id)
