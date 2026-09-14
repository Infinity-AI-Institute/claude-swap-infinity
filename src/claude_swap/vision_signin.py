"""Resumable browser consent using Vision's existing device-proof/S256 protocol."""

from __future__ import annotations

import base64
import hashlib
import math
import re
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from claude_swap.exceptions import SessionError
from claude_swap.vision import VisionError, VisionTransport, registry_id
from claude_swap.vision_state import VisionState


def _proof(value: Any) -> bool:
    return (
        isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{43}", value) is not None
    )


class VisionSignIn:
    def __init__(self, root: Path, url: str, *, transport=None, now=time.time):
        self.transport = transport or VisionTransport(url)
        self.url = self.transport.url
        self.state = VisionState(root)
        self.now = now

    def _pending(self) -> dict[str, Any] | None:
        value = self.state.read("pending")
        if value is None:
            return None
        fields = {
            "version",
            "url",
            "host_label",
            "verifier",
            "request_id",
            "device_secret",
            "comparison_code",
            "verification_url",
            "expires_at",
            "interval_seconds",
            "next_poll_at",
        }
        if (
            not isinstance(value, dict)
            or set(value) != fields
            or type(value["version"]) is not int
            or value["version"] != 1
            or value["url"] != self.url
            or not _proof(value["verifier"])
            or not isinstance(value["host_label"], str)
            or not 1 <= len(value["host_label"]) <= 100
            or type(value["next_poll_at"]) not in (int, float)
            or not math.isfinite(value["next_poll_at"])
        ):
            raise SessionError(
                "Pending Vision sign-in must be reconciled at its original origin."
            )
        self._validate_initiation({key: value[key] for key in self._response_fields()})
        return value

    @staticmethod
    def _response_fields() -> set[str]:
        return {
            "request_id",
            "device_secret",
            "comparison_code",
            "verification_url",
            "expires_at",
            "interval_seconds",
        }

    def _validate_initiation(self, value: Any) -> None:
        if not isinstance(value, dict) or set(value) != self._response_fields():
            raise VisionError("service_unavailable")
        registry_id(value["request_id"], "")
        if (
            not _proof(value["device_secret"])
            or not isinstance(value["comparison_code"], str)
            or not re.fullmatch(r"[0-9A-F]{10}", value["comparison_code"])
            or value["verification_url"]
            != self.url + "/cli-authorize/" + value["request_id"]
            or type(value["interval_seconds"]) is not int
            or not 1 <= value["interval_seconds"] <= 300
        ):
            raise VisionError("service_unavailable")
        try:
            stamp = datetime.fromisoformat(value["expires_at"])
            if stamp.utcoffset() is None:
                raise ValueError()
        except (ValueError, TypeError):
            raise VisionError("service_unavailable") from None

    @staticmethod
    def _public(value: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value[key]
            for key in (
                "request_id",
                "host_label",
                "comparison_code",
                "verification_url",
                "expires_at",
            )
        }

    def begin(self, host_label: str) -> dict[str, Any]:
        if not isinstance(host_label, str) or not 1 <= len(host_label.strip()) <= 100:
            raise SessionError("Use a host label between 1 and 100 characters.")
        with self.state.lock():
            pending = self._pending()
            if pending is not None:
                if pending["host_label"] != host_label.strip():
                    raise SessionError(
                        "Pending sign-in belongs to another host label; cancel it first."
                    )
                return self._public(pending)
            verifier = secrets.token_urlsafe(32)
            challenge = (
                base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
                .decode()
                .rstrip("=")
            )
            value = self.transport.request(
                "POST",
                "/api/cli-auth/requests",
                {
                    "client": "claude-swap",
                    "host_label": host_label.strip(),
                    "challenge": challenge,
                },
            )
            self._validate_initiation(value)
            expires = datetime.fromisoformat(value["expires_at"]).timestamp()
            if not self.now() < expires <= self.now() + 1200:
                raise VisionError("expired")
            pending = {
                "version": 1,
                "url": self.url,
                "host_label": host_label.strip(),
                "verifier": verifier,
                **value,
                "next_poll_at": self.now() + value["interval_seconds"],
            }
            self.state.write("pending", pending)
            return self._public(pending)

    def poll(self) -> dict[str, Any]:
        with self.state.lock():
            pending = self._pending()
            if pending is None:
                raise SessionError("Start Vision sign-in before polling for approval.")
            # Preserve the server's two-minute delivery recovery window after
            # request expiry; an issued response may have been lost at the boundary.
            expires = datetime.fromisoformat(pending["expires_at"]).timestamp() + 120
            if self.now() >= expires:
                raise VisionError("delivery_expired")
            remaining = pending["next_poll_at"] - self.now()
            if remaining > 0:
                return {"state": "pending", "retry_after_seconds": math.ceil(remaining)}
            pending["next_poll_at"] = self.now() + pending["interval_seconds"]
            self.state.write("pending", pending)
            try:
                value = self.transport.request(
                    "POST",
                    "/api/cli-auth/requests/" + pending["request_id"] + "/token",
                    {
                        "device_secret": pending["device_secret"],
                        "verifier": pending["verifier"],
                    },
                )
            except VisionError as error:
                if error.code not in {
                    "pending",
                    "slow_down",
                    "unavailable",
                    "service_unavailable",
                }:
                    raise
                delay = error.retry_after_seconds or pending["interval_seconds"]
                if error.code == "slow_down":
                    delay = max(delay, pending["interval_seconds"] + 5)
                pending["interval_seconds"] = min(
                    300, max(pending["interval_seconds"], delay)
                )
                pending["next_poll_at"] = self.now() + delay
                self.state.write("pending", pending)
                return {"state": "pending", "retry_after_seconds": delay}
            if (
                not isinstance(value, dict)
                or set(value) != {"api_key", "key_id"}
                or not isinstance(value["api_key"], str)
                or not re.fullmatch(r"vsk_[0-9a-f]{40}", value["api_key"])
            ):
                raise VisionError("service_unavailable")
            registry_id(value["key_id"], "vkey_")
            self.state.write(
                "key",
                {
                    "version": 1,
                    "url": self.url,
                    "request_id": pending["request_id"],
                    **value,
                },
            )
            self.state.remove("pending")
            return {"state": "signed_in", "url": self.url, "key_id": value["key_id"]}

    def cancel(self) -> dict[str, Any]:
        with self.state.lock():
            pending = self._pending()
            if pending is None:
                return {"state": "no_pending_request"}
            result = self.transport.request(
                "POST",
                "/api/cli-auth/requests/" + pending["request_id"] + "/cancel",
                {"device_secret": pending["device_secret"]},
            )
            if (
                not isinstance(result, dict)
                or set(result) != {"cancelled"}
                or result["cancelled"] is not True
            ):
                raise VisionError("service_unavailable")
            saved = self.state.read("key")
            if (
                isinstance(saved, dict)
                and saved.get("request_id") == pending["request_id"]
            ):
                self.state.remove("key")
            self.state.remove("pending")
            return {"state": "cancelled"}
