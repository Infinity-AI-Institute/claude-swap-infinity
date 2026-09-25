"""Vision registry protocol for centrally owned Claude subscription credentials.

This module performs bounded registry calls only. Provider refresh and native
credential storage remain outside this transport.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from claude_swap.exceptions import ClaudeSwitchError

MAX_RESPONSE_BYTES = 2 * 1024 * 1024
SUBSCRIPTION_FIELDS = {
    "plan",
    "rawPlan",
    "rawTier",
    "rawSeatTier",
    "multiplier",
    "source",
}
ERROR_CODES = {
    "unauthorized",
    "not_permitted",
    "unavailable",
    "credential_unavailable",
    "service_unavailable",
    "invalid_request",
    "request_conflict",
    "identity_mismatch",
    "invalid_credential",
    "unsupported",
    "verification_unavailable",
    "expired",
    "conflict",
    "slow_down",
    "rate_limited",
    "pending",
    "denied",
    "cancelled",
    "delivery_expired",
    "forbidden",
}


# The two refusals a key-only user can act on. A Vision API key is the only
# input they configured, so the fix is in that key or in the access Vision
# grants its owner, never a provider login. Other codes are service states.
ACCESS_REFUSAL_ADVICE = {
    "unauthorized": (
        "The Vision API key was not accepted. Create a new key in Vision "
        "Settings (gear) → API keys. If VISION_API_KEY is set, put the new key "
        "there, because it overrides a saved key. Otherwise run "
        "cswap --set-vision-token."
    ),
    "not_permitted": (
        "Your Vision user is not an invited member, or this API key lacks "
        "agent-account access. Ask a Vision admin."
    ),
}


class VisionError(ClaudeSwitchError):
    def __init__(
        self,
        code: str,
        status: int | None = None,
        retry_after_seconds: int | None = None,
        *,
        url: str | None = None,
    ):
        self.code = (
            code
            if isinstance(code, str) and code in ERROR_CODES
            else "service_unavailable"
        )
        self.status = status
        self.retry_after_seconds = retry_after_seconds
        source = "Vision account registry" if url is None else f"Vision at {url}"
        message = f"{source}: {self.code}."
        advice = ACCESS_REFUSAL_ADVICE.get(self.code)
        if advice is not None:
            message += " " + advice
        super().__init__(message)


def origin(value: str) -> str:
    try:
        if not isinstance(value, str):
            raise TypeError()
        parsed = urllib.parse.urlsplit(value)
        hostname = parsed.hostname or ""
        loopback = hostname == "localhost"
        if not loopback:
            try:
                loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                pass
        if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
            raise ValueError()
        if (
            not hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
            or parsed.port == 0
        ):
            raise ValueError()
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    except (ValueError, TypeError):
        raise VisionError("invalid_request") from None


def registry_id(value: Any, prefix: str) -> str:
    if not isinstance(value, str) or not value.startswith(prefix):
        raise VisionError("service_unavailable")
    try:
        parsed = uuid.UUID(value[len(prefix) :])
    except ValueError:
        raise VisionError("service_unavailable") from None
    if parsed.version != 4 or value != prefix + str(parsed):
        raise VisionError("service_unavailable")
    return value


def subscription(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VisionError("service_unavailable")
    result = {}
    for key in SUBSCRIPTION_FIELDS & value.keys():
        item = value[key]
        if item is not None and (
            type(item) not in (str, int, float) or len(str(item)) > 500
        ):
            raise VisionError("service_unavailable")
        if isinstance(item, float) and not math.isfinite(item):
            raise VisionError("service_unavailable")
        result[key] = item
    return result


def _identity(value: dict[str, Any]) -> None:
    registry_id(value.get("account_id"), "aia_")
    registry_id(value.get("login_id"), "ail_")
    if (
        value.get("provider") != "claude"
        or not isinstance(value.get("email"), str)
        or not 3 <= len(value["email"]) <= 320
        or not isinstance(value.get("organization_id"), str)
        or len(value["organization_id"]) > 500
    ):
        raise VisionError("service_unavailable")


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        fp.close()
        raise VisionError("service_unavailable", code)


class VisionTransport:
    """Bounded public transport; browser initiation sends no registry API key."""

    def __init__(self, url: str):
        self.url = origin(url)

    def _headers(self):
        return {"Accept": "application/json"}

    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        *,
        handoff_secret: str | None = None,
    ) -> Any:
        if not path.startswith("/api/") or any(c in path for c in "#\r\n"):
            raise VisionError("invalid_request")
        headers = self._headers()
        if handoff_secret is not None:
            if not isinstance(handoff_secret, str) or not re.fullmatch(
                r"[A-Za-z0-9_-]{43}", handoff_secret
            ):
                raise VisionError("invalid_request")
            headers["X-Vision-Handoff-Secret"] = handoff_secret
        payload = None
        if body is not None:
            payload = json.dumps(body, allow_nan=False).encode()
            if len(payload) > MAX_RESPONSE_BYTES:
                raise VisionError("invalid_request")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            self.url + path, data=payload, headers=headers, method=method
        )
        try:
            response = urllib.request.build_opener(NoRedirects()).open(req, timeout=10)
        except urllib.error.HTTPError as error:
            with error:
                try:
                    raw = json.loads(error.read(4096))
                    code = raw.get("error", {}).get("code")
                except (ValueError, AttributeError, OSError):
                    code = "service_unavailable"
            retry_after = None
            raw_retry = error.headers.get("Retry-After", "") if error.headers else ""
            if (
                len(raw_retry) <= 4
                and raw_retry.isascii()
                and raw_retry.isdigit()
                and 1 <= int(raw_retry) <= 3600
            ):
                retry_after = int(raw_retry)
            raise VisionError(code, error.code, retry_after, url=self.url) from None
        except VisionError:
            raise
        except (OSError, urllib.error.URLError, ValueError):
            raise VisionError("service_unavailable") from None
        try:
            with response:
                chunks, total = [], 0
                deadline = time.monotonic() + 10
                while True:
                    if time.monotonic() >= deadline:
                        raise VisionError("service_unavailable")
                    part = response.read1(min(65536, MAX_RESPONSE_BYTES + 1 - total))
                    if not part:
                        break
                    total += len(part)
                    if total > MAX_RESPONSE_BYTES:
                        raise VisionError("service_unavailable")
                    chunks.append(part)
            raw = json.loads(b"".join(chunks))
            if not isinstance(raw, dict) or set(raw) != {"data"}:
                raise VisionError("service_unavailable")
            return raw["data"]
        except (OSError, ValueError):
            raise VisionError("service_unavailable") from None


@dataclass(repr=False)
class VisionClient(VisionTransport):
    url: str
    api_key: str = field(repr=False)

    def __post_init__(self):
        self.url = origin(self.url)
        if (
            not isinstance(self.api_key, str)
            or not 1 <= len(self.api_key) <= 4096
            or any(c in self.api_key for c in "\r\n")
        ):
            raise VisionError("unauthorized")

    @property
    def scope(self) -> str:
        """Nonsecret stable namespace; never use a local alias as provider identity."""
        return hashlib.sha256(self.url.encode()).hexdigest()

    def _headers(self):
        return {"Accept": "application/json", "X-Api-Key": self.api_key}

    def discover(self) -> list[dict[str, Any]]:
        result, cursor = [], None
        while True:
            query = {"version": "1", "provider": "claude", "limit": "100"}
            if cursor:
                query["cursor"] = cursor
            page = self.request(
                "GET", "/api/ai-accounts/registry?" + urllib.parse.urlencode(query)
            )
            if (
                not isinstance(page, dict)
                or set(page) != {"version", "items", "next_cursor"}
                or type(page["version"]) is not int
                or page["version"] != 1
                or not isinstance(page["items"], list)
                or len(page["items"]) > 100
            ):
                raise VisionError("service_unavailable")
            previous = cursor
            for item in page["items"]:
                if not isinstance(item, dict):
                    raise VisionError("service_unavailable")
                _identity(item)
                login_id = item["login_id"]
                if previous is not None and login_id <= previous:
                    raise VisionError("service_unavailable")
                generation = item.get("login_generation")
                if generation is not None and (
                    type(generation) is not int or generation < 1
                ):
                    raise VisionError("service_unavailable")
                title = item.get("title")
                if title is not None and (
                    not isinstance(title, str) or len(title) > 200
                ):
                    raise VisionError("service_unavailable")
                result.append(
                    {
                        key: item[key]
                        for key in (
                            "account_id",
                            "login_id",
                            "provider",
                            "email",
                            "organization_id",
                        )
                    }
                    | {
                        "title": title,
                        "login_generation": generation,
                        "subscription": subscription(item.get("subscription")),
                    }
                )
                previous = login_id
            following = page["next_cursor"]
            if following is None:
                return result
            registry_id(following, "ail_")
            if (
                not page["items"]
                or following != previous
                or (cursor and following <= cursor)
            ):
                raise VisionError("service_unavailable")
            cursor = following

    def credential(self, account_id: str, login_id: str) -> dict[str, Any]:
        registry_id(account_id, "aia_")
        registry_id(login_id, "ail_")
        value = self.request(
            "POST",
            f"/api/ai-accounts/{account_id}/logins/{login_id}/credentials",
            {"version": 1},
        )
        fields = {
            "version",
            "account_id",
            "login_id",
            "provider",
            "email",
            "organization_id",
            "kind",
            "generation",
            "accessToken",
            "subscription",
            "capabilities",
            "verified_at",
            "expires_at",
        }
        if (
            not isinstance(value, dict)
            or set(value) != fields
            or type(value["version"]) is not int
            or value["version"] != 1
            or value["account_id"] != account_id
            or value["login_id"] != login_id
        ):
            raise VisionError("service_unavailable")
        _identity(value)
        if (
            value["kind"] not in ("login_oauth", "subscription_oauth_token")
            or type(value["generation"]) is not int
            or value["generation"] < 1
            or not isinstance(value["accessToken"], str)
            or not 1 <= len(value["accessToken"]) <= 65536
            or any(c in value["accessToken"] for c in "\r\n")
        ):
            raise VisionError("service_unavailable")
        capabilities = value["capabilities"]
        if (
            not isinstance(capabilities, list)
            or len(capabilities) > 256
            or any(
                not isinstance(item, str) or len(item) > 100 for item in capabilities
            )
        ):
            raise VisionError("service_unavailable")
        try:
            verified = datetime.fromisoformat(value["verified_at"])
            if verified.utcoffset() is None:
                raise ValueError()
            if value["expires_at"] is not None:
                expiry = datetime.fromisoformat(value["expires_at"])
                if expiry.utcoffset() is None:
                    raise ValueError()
                if expiry.timestamp() <= time.time() + 30:
                    raise VisionError("credential_unavailable")
        except (ValueError, TypeError):
            raise VisionError("service_unavailable") from None
        return {**value, "subscription": subscription(value["subscription"])}

    def refresh(
        self, account_id: str, login_id: str, generation: int
    ) -> dict[str, Any]:
        registry_id(account_id, "aia_")
        registry_id(login_id, "ail_")
        if type(generation) is not int or not 1 <= generation <= 2147483646:
            raise VisionError("invalid_request")
        value = self.request(
            "POST",
            f"/api/ai-accounts/{account_id}/logins/{login_id}/refresh",
            {"version": 1, "generation": generation},
        )
        if (
            not isinstance(value, dict)
            or type(value.get("version")) is not int
            or value["version"] != 1
            or type(value.get("generation")) is not int
            or value["generation"] < 1
            or not isinstance(value.get("state"), str)
            or value.get("state")
            not in {
                "current",
                "superseded",
                "queued",
                "running",
                "completed",
                "reauth_required",
                "obsolete",
            }
        ):
            raise VisionError("service_unavailable")
        return value


def configured_client(state_root: Path | None = None) -> VisionClient | None:
    """Resolve environment, shared manual token, then this installation's sign-in."""
    from claude_swap.paths import get_backup_root
    from claude_swap.vision_state import VisionState

    key = os.environ.get("VISION_API_KEY")
    if key:
        return VisionClient(
            os.environ.get("VISION_API_URL", "https://vision.infinity.inc"), key
        )
    from claude_swap.vision_token import saved_token_client

    shared = saved_token_client()
    if shared is not None:
        return shared
    saved = VisionState(
        state_root if state_root is not None else get_backup_root()
    ).read("key")
    if saved is None:
        return None
    if (
        not isinstance(saved, dict)
        or set(saved) != {"version", "url", "request_id", "key_id", "api_key"}
        or type(saved["version"]) is not int
        or saved["version"] != 1
        or not isinstance(saved["api_key"], str)
        or not re.fullmatch(r"vsk_[0-9a-f]{40}", saved["api_key"])
    ):
        raise VisionError("invalid_request")
    registry_id(saved["request_id"], "")
    registry_id(saved["key_id"], "vkey_")
    saved_url = origin(saved["url"])
    override = os.environ.get("VISION_API_URL")
    if override and origin(override) != saved_url:
        from claude_swap.exceptions import SessionError

        raise SessionError(
            "Saved Vision sign-in belongs to another origin; sign in there separately."
        )
    return VisionClient(saved_url, saved["api_key"])
