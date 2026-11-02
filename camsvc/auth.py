"""Log in with passkeys (WebAuthn), without passwords.

- Passkeys only work over HTTPS with a domain name (e.g. https://camera.example.com behind a
  reverse proxy). rp_id is that domain name, origin the full https URL.
- The first passkey is paired with a one-time pairing code (in the log and in
  <state_dir>/pairing_code). After that, add more devices while logged in; with iCloud Keychain
  a passkey created on a Mac is immediately available on the iPhone too.
- After logging in, a signed cookie (HttpOnly, Secure) stays valid for `session_days`.
  "Log out everywhere" replaces the secret, which invalidates all cookies.
- Requests directly from the server itself (127.0.0.1, not via the proxy) need no login: handy
  for checks and scripts on the server. A reverse proxy always sets X-Forwarded-For, so it does
  not count.
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

COOKIE = "camsvc_session"
CHALLENGE_TTL = 300.0
USER_ID = b"camera-service-huishouden"   # opaque id of the one shared account (kept for existing passkeys)


class AuthError(Exception):
    pass


class Auth:
    def __init__(self, state_dir: Path, rp_id: str, origin: str, session_days: float = 90):
        self.rp_id = rp_id
        self.origin = origin
        self.session_seconds = int(session_days * 86400)
        self._file = state_dir / "auth.json"
        self._code_file = state_dir / "pairing_code"
        self._lock = threading.Lock()
        self._challenges: dict[str, tuple[bytes, float, str]] = {}   # id -> (challenge, until, kind)
        data = self._load()
        if not data.get("secret"):
            data["secret"] = secrets.token_hex(32)
            self._save(data)
        if not data.get("credentials") and not self._code_file.exists():
            self.new_setup_code()       # an existing code stays valid across restarts

    # ------------------------------------------------------------------ storage
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
        return [{k: c[k] for k in ("id", "name", "created", "last_used")}
                for c in self._load().get("credentials", [])]

    def new_setup_code(self) -> str:
        code = "-".join(secrets.token_hex(2).upper() for _ in range(3))
        self._code_file.write_text(code + "\n")
        self._code_file.chmod(0o600)
        log.warning("Passkey pairing code: %s (one-time, also in %s)", code, self._code_file)
        return code

    def _setup_code_ok(self, code: str | None) -> bool:
        try:
            expected = self._code_file.read_text().strip()
        except FileNotFoundError:
            return False
        return bool(code) and hmac.compare_digest(expected.encode(), code.strip().upper().encode())

    # ------------------------------------------------------------------ sessions
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
        # a removed passkey also logs that device out
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
            raise AuthError("Expired, please try again")
        return item[0]

    # ------------------------------------------------------------------ registration
    def registration_options(self, logged_in: bool, setup_code: str | None) -> dict:
        if not logged_in and not self._setup_code_ok(setup_code):
            raise AuthError("Pairing code is not correct")
        existing = self._load().get("credentials", [])
        opts = generate_registration_options(
            rp_id=self.rp_id, rp_name="Uitkijk", user_id=USER_ID, user_name="household",
            user_display_name="Uitkijk cameras",
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
            raise AuthError("Pairing code is not correct")
        challenge = self._take_challenge(challenge_id, "reg")
        try:
            v = verify_registration_response(credential=credential, expected_challenge=challenge,
                                             expected_rp_id=self.rp_id, expected_origin=self.origin,
                                             require_user_verification=True)
        except Exception as exc:  # noqa: BLE001 - the library raises various error types
            raise AuthError(f"Passkey not accepted: {exc}") from exc
        cred_id = bytes_to_base64url(v.credential_id)
        today = time.strftime("%Y-%m-%d")
        with self._lock:
            data = self._load()
            data.setdefault("credentials", []).append({
                "id": cred_id, "public_key": bytes_to_base64url(v.credential_public_key),
                "sign_count": v.sign_count, "name": (name or "Device").strip()[:40],
                "created": today, "last_used": today})
            self._save(data)
        self._code_file.unlink(missing_ok=True)       # the pairing code is single-use
        log.info("Passkey added: %s", name)
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

    # ------------------------------------------------------------------ login
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
                raise AuthError("Unknown passkey")
            try:
                v = verify_authentication_response(
                    credential=credential, expected_challenge=challenge, expected_rp_id=self.rp_id,
                    expected_origin=self.origin,
                    credential_public_key=base64url_to_bytes(stored["public_key"]),
                    credential_current_sign_count=stored["sign_count"], require_user_verification=True)
            except Exception as exc:  # noqa: BLE001
                raise AuthError("Login failed") from exc
            stored["sign_count"] = v.new_sign_count
            stored["last_used"] = time.strftime("%Y-%m-%d")
            self._save(data)
        return self.make_cookie(cred_id)
