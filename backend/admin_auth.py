"""Short-lived signed administrator sessions for browser and API clients."""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
import time
from typing import Iterable
from urllib.parse import urlsplit


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def issue_session_token(user: str, signing_key: str, ttl_seconds: int, now: int | None = None) -> str:
    issued_at = int(time.time()) if now is None else int(now)
    payload = json.dumps(
        {
            "sub": user,
            "exp": issued_at + int(ttl_seconds),
            "nonce": secrets.token_urlsafe(16),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    encoded = _encode(payload)
    signature = hmac.new(signing_key.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256).digest()
    return f"{encoded}.{_encode(signature)}"


def verify_session_token(
    token: str,
    expected_user: str,
    signing_key: str,
    now: int | None = None,
) -> bool:
    current = int(time.time()) if now is None else int(now)
    try:
        encoded, signature = token.split(".", 1)
        expected = hmac.new(
            signing_key.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
        ).digest()
        if not secrets.compare_digest(_decode(signature), expected):
            return False
        payload = json.loads(_decode(encoded))
        return payload.get("sub") == expected_user and int(payload.get("exp") or 0) > current
    except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError, binascii.Error):
        return False


def _normalized_origin(value: str) -> str:
    try:
        parsed = urlsplit(str(value or "").strip())
    except ValueError:
        return ""
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def browser_origin_allowed(
    origin: str,
    referer: str,
    expected_origin: str,
    allowed_origins: Iterable[str] = (),
) -> bool:
    """Validate the browser source for cookie-authenticated write requests."""
    source = _normalized_origin(origin) or _normalized_origin(referer)
    if not source:
        return False
    trusted = {_normalized_origin(expected_origin)}
    trusted.update(_normalized_origin(item) for item in allowed_origins)
    trusted.discard("")
    if source in trusted:
        return True
    source_parts = urlsplit(source)
    return any(
        source_parts.hostname == urlsplit(candidate).hostname
        for candidate in trusted
    )


__all__ = ["browser_origin_allowed", "issue_session_token", "verify_session_token"]
