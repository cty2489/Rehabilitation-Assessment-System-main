"""Parse and verify legacy and per-device API credentials."""
from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from typing import Dict, Optional


class DeviceTokenConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DeviceCredential:
    device_id: Optional[str]
    legacy: bool = False


def generate_device_token() -> str:
    return secrets.token_urlsafe(32)


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_hint(token: str) -> str:
    value = token.strip()
    if len(value) <= 12:
        return value
    return f"{value[:6]}...{value[-6:]}"


def parse_named_tokens(raw: str) -> Dict[str, str]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DeviceTokenConfigError("DEVICE_API_TOKENS_JSON 不是合法 JSON") from exc
    if not isinstance(payload, dict):
        raise DeviceTokenConfigError("DEVICE_API_TOKENS_JSON 必须是 device_id 到 token 的对象")

    result: Dict[str, str] = {}
    seen_tokens = set()
    for raw_device_id, raw_token in payload.items():
        device_id = str(raw_device_id or "").strip()
        token = str(raw_token or "").strip()
        if not device_id or not token:
            raise DeviceTokenConfigError("设备 ID 和 token 不能为空")
        if token in seen_tokens:
            raise DeviceTokenConfigError("不同设备不能配置相同 token")
        seen_tokens.add(token)
        result[device_id] = token
    return result


def credential_count(legacy_token: str, named_tokens_json: str) -> int:
    named = parse_named_tokens(named_tokens_json)
    legacy = (legacy_token or "").strip()
    if legacy and legacy in named.values():
        raise DeviceTokenConfigError("旧 DEVICE_API_TOKEN 不能与独立设备 token 重复")
    return len(named) + (1 if legacy else 0)


def authenticate_device_token(
    provided_token: str,
    legacy_token: str,
    named_tokens_json: str,
) -> Optional[DeviceCredential]:
    named = parse_named_tokens(named_tokens_json)
    legacy = (legacy_token or "").strip()
    provided = (provided_token or "").strip()
    if legacy and legacy in named.values():
        raise DeviceTokenConfigError("旧 DEVICE_API_TOKEN 不能与独立设备 token 重复")
    if not provided:
        return None
    if legacy and secrets.compare_digest(provided, legacy):
        return DeviceCredential(device_id=None, legacy=True)
    for device_id, expected in named.items():
        if secrets.compare_digest(provided, expected):
            return DeviceCredential(device_id=device_id, legacy=False)
    return None


__all__ = [
    "DeviceCredential",
    "DeviceTokenConfigError",
    "authenticate_device_token",
    "credential_count",
    "generate_device_token",
    "parse_named_tokens",
    "token_digest",
    "token_hint",
]
