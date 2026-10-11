"""Best-effort usage/quota polling for Codex CLI accounts.

Every fetch here is defensive by contract: a caller must always be able to
degrade to "unavailable" instead of crashing. The fetcher raises
:class:`CodexUsageError` with a short reason ("http-401", "timeout", ...) on
any failure and never leaks tokens into messages. Ported from the
``multi-tool-runner`` branch's ``tool_usage.py``, Codex parts only.

Contract (verified against codex's own client): Codex CLI reads
``GET https://chatgpt.com/backend-api/wham/usage`` with the OAuth access
token as a Bearer token and the account id in the ``ChatGPT-Account-Id``
header. The response carries ``plan_type`` and
``rate_limit.{primary,secondary}_window`` with ``used_percent`` /
``reset_at`` / ``limit_window_seconds`` (primary = 5h, secondary = weekly).

:func:`as_usage_dict` renders a fetched result in the same normalized shape
``oauth.build_usage_result`` produces for Claude accounts, so the existing
wall machinery (``oauth.switch_margin`` with ``autoswitch.threshold`` /
``autoswitch.thresholds`` per-window walls) evaluates Codex accounts without
duplication — Codex's 5h/weekly windows land on the same "5h"/"7d" labels
the walls already understand.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

from claude_swap.codex_accounts import _decode_jwt_payload

CODEX_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"

_TIMEOUT_S = 10.0


class CodexUsageError(Exception):
    """A usage fetch failed; the message is safe to display to the user."""


@dataclass(frozen=True)
class UsageWindow:
    """One quota window: label, % used, and when it resets (epoch, if known)."""

    label: str
    pct: float
    resets_at: float | None = None


@dataclass(frozen=True)
class CodexUsage:
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
        raise CodexUsageError(_classify_url_error(e)) from e
    try:
        data = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise CodexUsageError("bad-response") from e
    if not isinstance(data, dict):
        raise CodexUsageError("bad-response")
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
    if not isinstance(pct, (int, float)) or isinstance(pct, bool):
        return None
    reset_at = data.get("reset_at", data.get("resetAt"))
    resets = float(reset_at) if isinstance(reset_at, (int, float)) else None
    seconds = data.get("limit_window_seconds", data.get("limitWindowSeconds"))
    label = _window_label(seconds if isinstance(seconds, int) else None, fallback_label)
    return UsageWindow(label=label, pct=float(pct), resets_at=resets)


def fetch_codex_usage(credential: dict) -> CodexUsage:
    """Fetch ChatGPT rate-limit usage for a Codex auth.json credential."""
    tokens = credential.get("tokens") or {}
    access_token = tokens.get("access_token")
    if not access_token:
        raise CodexUsageError("no-access-token")
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
        raise CodexUsageError("no-rate-limits")
    plan = data.get("plan_type") or data.get("planType") or ""
    return CodexUsage(plan=str(plan), windows=tuple(windows))


def fetch_usage_from_text(credential_text: str) -> CodexUsage:
    """Fetch usage for a raw auth.json snapshot (slot file content)."""
    try:
        credential = json.loads(credential_text)
    except json.JSONDecodeError as e:
        raise CodexUsageError("bad-credential") from e
    if not isinstance(credential, dict):
        raise CodexUsageError("bad-credential")
    return fetch_codex_usage(credential)


def _iso_from_epoch(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return (
        datetime.fromtimestamp(epoch, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def as_usage_dict(usage: CodexUsage) -> dict:
    """Render a Codex usage result in ``oauth.build_usage_result``'s shape.

    The "5h" window becomes ``five_hour`` and "7d" becomes ``seven_day`` —
    the labels ``oauth.relevant_windows`` re-emits, so the same
    ``autoswitch.thresholds`` walls ("5h=95,7d=98") gate Codex accounts
    identically. A window with any other label (the API reporting an unusual
    ``limit_window_seconds``) is folded in as a ``scoped`` entry under its
    own name rather than dropped: evaluated with ``models=("all",)``, every
    window the account reports gates it — an unrecognized window must never
    be invisible to the walls.
    """
    result: dict = {}
    scoped: list[dict] = []
    for window in usage.windows:
        entry: dict = {"pct": window.pct}
        resets_at = _iso_from_epoch(window.resets_at)
        if resets_at is not None:
            entry["resets_at"] = resets_at
        if window.label == "5h":
            result["five_hour"] = entry
        elif window.label == "7d":
            result["seven_day"] = entry
        else:
            scoped.append({"name": window.label, **entry})
    if scoped:
        result["scoped"] = scoped
    return result
