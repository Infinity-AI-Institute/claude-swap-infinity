"""Best-effort usage/quota polling for the non-Claude tools (Codex, Kimi).

Every fetch here is defensive by contract: a caller must always be able to
degrade to "unavailable" instead of crashing. Both fetchers raise
:class:`ToolUsageError` with a short reason ("http-401", "timeout", ...) on
any failure and never leak tokens into messages.

Contracts (verified against the tools' own clients):

- Codex CLI reads ``GET https://chatgpt.com/backend-api/wham/usage`` with the
  OAuth access token as a Bearer token and the account id in the
  ``ChatGPT-Account-Id`` header. The response carries ``plan_type`` and
  ``rate_limit.{primary,secondary}_window`` with ``used_percent`` /
  ``reset_at`` / ``limit_window_seconds`` (primary = 5h, secondary = weekly).
- Kimi Code reads ``GET https://api.kimi.com/coding/v1/usages`` with the
  OAuth access token plus the ``X-Msh-*`` device headers the official CLI
  sends (the backend 403s a foreign User-Agent). Response rows come from
  ``usage`` (weekly) and ``limits[]`` with ``used``/``limit``/``remaining``
  and reset hints (``reset_at`` ISO, or ``reset_in``/``ttl`` seconds).
"""

from __future__ import annotations

import json
import os
import platform
import socket
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from claude_swap.tool_switcher import _decode_jwt_payload

CODEX_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
KIMI_USAGE_URL = "https://api.kimi.com/coding/v1/usages"

# Mirrors kimi-cli v1.41.0 verbatim: Moonshot's `kimi-for-coding` backend 403s
# on any other User-Agent prefix ("access_terminated_error: only available
# for Coding Agents"). This is a public constant shipped inside the CLI.
KIMI_CLI_VERSION = "1.41.0"
KIMI_USER_AGENT = f"KimiCLI/{KIMI_CLI_VERSION}"

_TIMEOUT_S = 10.0


class ToolUsageError(Exception):
    """A usage fetch failed; the message is safe to display to the user."""


@dataclass(frozen=True)
class UsageWindow:
    """One quota window: label, % used, and when it resets (epoch, if known)."""

    label: str
    pct: float
    resets_at: float | None = None


@dataclass(frozen=True)
class ToolUsage:
    """Parsed usage for one account; ``windows`` may be empty on odd payloads."""

    plan: str = ""
    windows: tuple[UsageWindow, ...] = field(default_factory=tuple)


def _classify_url_error(e: BaseException) -> str:
    """Short failure tag mirroring oauth._classify's categories."""
    if isinstance(e, urllib.error.HTTPError):
        return f"http-{e.code}"
    if isinstance(e, (TimeoutError, socket.timeout)):
        return "timeout"
    if isinstance(e, urllib.error.URLError):
        if isinstance(e.reason, (TimeoutError, socket.timeout)):
            return "timeout"
        return "network"
    return "bad-response"


def _get_json(url: str, headers: dict[str, str]) -> dict:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            body = resp.read(1 << 20)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ToolUsageError(_classify_url_error(e)) from e
    try:
        data = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ToolUsageError("bad-response") from e
    if not isinstance(data, dict):
        raise ToolUsageError("bad-response")
    return data


def _window_label(window_seconds: int | None, fallback: str) -> str:
    """Human label for a window length: 18000s -> '5h', 604800s -> '7d'."""
    if not window_seconds or window_seconds <= 0:
        return fallback
    if window_seconds % 86400 == 0:
        return f"{window_seconds // 86400}d"
    if window_seconds % 3600 == 0:
        return f"{window_seconds // 3600}h"
    if window_seconds % 60 == 0:
        return f"{window_seconds // 60}m"
    return fallback


def _parse_window(data: object, fallback_label: str) -> UsageWindow | None:
    if not isinstance(data, dict):
        return None
    pct = data.get("used_percent", data.get("usedPercent"))
    if not isinstance(pct, (int, float)):
        return None
    reset_at = data.get("reset_at", data.get("resetAt"))
    resets = float(reset_at) if isinstance(reset_at, (int, float)) else None
    seconds = data.get("limit_window_seconds", data.get("limitWindowSeconds"))
    label = _window_label(seconds if isinstance(seconds, int) else None, fallback_label)
    return UsageWindow(label=label, pct=float(pct), resets_at=resets)


# -- Codex ---------------------------------------------------------------------


def fetch_codex_usage(credential: dict) -> ToolUsage:
    """Fetch ChatGPT rate-limit usage for a Codex auth.json credential."""
    tokens = credential.get("tokens") or {}
    access_token = tokens.get("access_token")
    if not access_token:
        raise ToolUsageError("no-access-token")
    account_id = tokens.get("account_id") or ""
    if not account_id:
        claims = _decode_jwt_payload(tokens.get("id_token") or "")
        auth_claims = claims.get("https://api.openai.com/auth") or {}
        account_id = auth_claims.get("chatgpt_account_id") or ""

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "User-Agent": "cswap",
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    data = _get_json(CODEX_USAGE_URL, headers)

    rate_limit = data.get("rate_limit") or data.get("rateLimit") or {}
    windows = []
    primary = _parse_window(rate_limit.get("primary_window"), "5h")
    if primary is not None:
        windows.append(primary)
    secondary = _parse_window(rate_limit.get("secondary_window"), "7d")
    if secondary is not None:
        windows.append(secondary)
    if not windows:
        raise ToolUsageError("no-rate-limits")
    plan = data.get("plan_type") or data.get("planType") or ""
    return ToolUsage(plan=str(plan), windows=tuple(windows))


# -- Kimi ----------------------------------------------------------------------


def _kimi_device_id() -> str:
    """The kimi-cli device fingerprint (shared at ~/.kimi/device_id).

    Mirrors kimi-cli's get_device_id: a persisted UUIDv4 hex (no dashes) so
    the usage request is indistinguishable from the real CLI's. Best-effort —
    an unreadable/unwritable path falls back to an ephemeral id rather than
    failing the fetch.
    """
    path = Path.home() / ".kimi" / "device_id"
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except OSError:
        pass
    device_id = uuid.uuid4().hex
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(device_id, encoding="utf-8")
    except OSError:
        pass
    return device_id


def _kimi_headers(access_token: str) -> dict[str, str]:
    """The headers kimi-cli sends on every request (see opencode-kimi-full
    src/headers.ts — deviations cause the backend to 403)."""
    system = platform.system() or "Unknown"
    release = platform.release()
    machine = platform.machine()
    if system == "Windows":
        build = release.split(".")[-1]
        label = "11" if build.isdigit() and int(build) >= 22000 else "10"
        model = f"Windows {label} {machine}".strip()
    else:
        model = f"{system} {release} {machine}".strip()
    return {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "User-Agent": KIMI_USER_AGENT,
        "X-Msh-Platform": "kimi_cli",
        "X-Msh-Version": KIMI_CLI_VERSION,
        "X-Msh-Device-Name": socket.gethostname() or "unknown",
        "X-Msh-Device-Model": model,
        "X-Msh-Device-Id": _kimi_device_id(),
        "X-Msh-Os-Version": f"{system} {release}".strip(),
    }


def _as_int(value: object) -> int | None:
    # The live /usages payload serializes numbers as strings ("limit": "100").
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _parse_reset(data: dict) -> float | None:
    """Reset epoch from a row: ISO ``reset_at``/``resetTime`` or relative
    ``reset_in``/``ttl`` seconds."""
    import time

    raw = (
        data.get("reset_at")
        or data.get("resetAt")
        or data.get("reset_time")
        or data.get("resetTime")
    )
    if isinstance(raw, str) and raw:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return dt.timestamp()
        except ValueError:
            return None
    seconds = _as_int(data.get("reset_in") or data.get("resetIn") or data.get("ttl"))
    if seconds is not None:
        return time.time() + seconds
    return None


def _kimi_row(data: dict, default_label: str) -> UsageWindow | None:
    limit = _as_int(data.get("limit"))
    used = _as_int(data.get("used"))
    remaining = _as_int(data.get("remaining"))
    if used is None and remaining is not None and limit is not None:
        used = limit - remaining
    if used is None or not limit:
        return None
    label = data.get("name") or data.get("title") or default_label
    pct = min(100.0, max(0.0, used / limit * 100))
    return UsageWindow(label=str(label), pct=pct, resets_at=_parse_reset(data))


def _kimi_limit_label(item: dict, detail: dict, idx: int) -> str:
    explicit = (
        item.get("name")
        or item.get("title")
        or item.get("scope")
        or detail.get("name")
        or detail.get("title")
        or detail.get("scope")
    )
    if explicit:
        return str(explicit)
    window = item.get("window") if isinstance(item.get("window"), dict) else {}
    duration = _as_int(window.get("duration") or item.get("duration") or detail.get("duration"))
    unit = str(window.get("timeUnit") or item.get("timeUnit") or detail.get("timeUnit") or "")
    if duration:
        if "MINUTE" in unit:
            return f"{duration // 60}h limit" if duration % 60 == 0 else f"{duration}m limit"
        if "HOUR" in unit:
            return f"{duration}h limit"
        if "DAY" in unit:
            return f"{duration}d limit"
        return f"{duration}s limit"
    return f"Limit #{idx + 1}"


def fetch_kimi_usage(credential: dict) -> ToolUsage:
    """Fetch subscription usage for a Kimi Code kimi-code.json credential."""
    access_token = credential.get("access_token")
    if not access_token:
        raise ToolUsageError("no-access-token")
    data = _get_json(KIMI_USAGE_URL, _kimi_headers(access_token))

    windows: list[UsageWindow] = []
    usage = data.get("usage")
    if isinstance(usage, dict):
        row = _kimi_row(usage, "Weekly limit")
        if row is not None:
            windows.append(row)
    limits = data.get("limits")
    if isinstance(limits, list):
        for idx, item in enumerate(limits):
            if not isinstance(item, dict):
                continue
            detail = item.get("detail") if isinstance(item.get("detail"), dict) else item
            row = _kimi_row(detail, _kimi_limit_label(item, detail, idx))
            if row is not None:
                windows.append(row)
    if not windows:
        raise ToolUsageError("no-usage-rows")
    # Membership tier, e.g. user.membership.level == "LEVEL_STANDARD".
    level = ((data.get("user") or {}).get("membership") or {}).get("level") or ""
    plan = str(level).removeprefix("LEVEL_").lower() if level else ""
    return ToolUsage(plan=plan, windows=tuple(windows))


def fetch_usage(tool: str, credential_text: str) -> ToolUsage:
    """Dispatch by tool key; ``credential_text`` is the raw slot file content."""
    try:
        credential = json.loads(credential_text)
    except json.JSONDecodeError as e:
        raise ToolUsageError("bad-credential") from e
    if not isinstance(credential, dict):
        raise ToolUsageError("bad-credential")
    if tool == "codex":
        return fetch_codex_usage(credential)
    if tool == "kimi":
        return fetch_kimi_usage(credential)
    raise ToolUsageError(f"unknown tool '{tool}'")
