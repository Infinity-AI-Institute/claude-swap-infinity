"""Claude registration receipts and two-phase handoff transport.

The caller either fences local refresh writers or explicitly acknowledges live
refresh risk. This transport validates that choice, not process quiescence.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from claude_swap.vision import VisionClient, VisionError, registry_id


def registration_proof(request_id: str, proof: str) -> None:
    try:
        registry_id(request_id, "")
    except VisionError:
        raise VisionError("invalid_request") from None
    if not isinstance(proof, str) or not re.fullmatch(r"[A-Za-z0-9_-]{43}", proof):
        raise VisionError("invalid_request")


def registration_receipt(value: Any, request_id: str) -> dict[str, Any]:
    fields = {
        "version",
        "request_id",
        "state",
        "account_id",
        "login_id",
        "provider",
        "email",
        "organization_id",
        "kind",
        "expected_generation",
        "generation",
        "expires_at",
        "capabilities",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or type(value.get("version")) is not int
        or value["version"] != 1
        or value.get("request_id") != request_id
        or value.get("provider") != "claude"
        or value.get("kind") not in ("login_oauth", "subscription_oauth_token")
    ):
        raise VisionError("service_unavailable")
    registry_id(value["request_id"], "")
    state = value["state"]
    if not isinstance(state, str) or state not in {
        "preparing",
        "pending_handoff",
        "committed",
        "cancelled",
        "expired",
        "conflict",
    }:
        raise VisionError("service_unavailable")
    registry_id(value["account_id"], "aia_")
    registry_id(value["login_id"], "ail_")
    expected = value["expected_generation"]
    generation = value["generation"]
    if (
        type(expected) is not int
        or not 0 <= expected < 2147483647
        or (
            generation is not None
            and (type(generation) is not int or generation != expected + 1)
        )
        or (state == "committed" and generation is None)
    ):
        raise VisionError("service_unavailable")
    if (
        not isinstance(value["email"], str)
        or not 3 <= len(value["email"]) <= 320
        or not isinstance(value["organization_id"], str)
        or not len(value["organization_id"]) <= 500
        or not isinstance(value["capabilities"], list)
        or len(value["capabilities"]) > 256
        or any(
            not isinstance(item, str) or len(item) > 100
            for item in value["capabilities"]
        )
    ):
        raise VisionError("service_unavailable")
    try:
        if datetime.fromisoformat(value["expires_at"]).utcoffset() is None:
            raise ValueError()
    except (TypeError, ValueError):
        raise VisionError("service_unavailable") from None
    return value


class RegistrationClient:
    def __init__(self, client: VisionClient):
        self.client = client

    def prepare_registration(
        self,
        request_id: str,
        proof: str,
        credential: dict[str, str],
        *,
        kind: str = "login_oauth",
    ) -> dict[str, Any]:
        registration_proof(request_id, proof)
        if kind not in ("login_oauth", "subscription_oauth_token"):
            raise VisionError("invalid_request")
        fields = (
            {"accessToken", "refreshToken"}
            if kind == "login_oauth"
            else {"accessToken"}
        )
        if not isinstance(credential, dict) or set(credential) != fields:
            raise VisionError("invalid_request")
        for value in credential.values():
            if (
                not isinstance(value, str)
                or not 1 <= len(value) <= 65536
                or any(character in value for character in "\r\n\0")
            ):
                raise VisionError("invalid_request")
        # Alias and cached identity cannot select the registration target. Vision
        # verifies this exact credential, including every re-login under an alias.
        value = self.client.request(
            "POST",
            "/api/ai-accounts/registrations",
            {
                "version": 1,
                "request_id": request_id,
                "handoff_secret": proof,
                "provider": "claude",
                "kind": kind,
                "credential": credential,
            },
        )
        receipt = registration_receipt(value, request_id)
        if receipt["kind"] != kind:
            raise VisionError("service_unavailable")
        return receipt

    def registration_status(self, request_id: str, proof: str) -> dict[str, Any]:
        registration_proof(request_id, proof)
        value = self.client.request(
            "GET",
            f"/api/ai-accounts/registrations/{request_id}",
            handoff_secret=proof,
        )
        return registration_receipt(value, request_id)

    def confirm_registration(
        self,
        request_id: str,
        proof: str,
        *,
        local_refreshers_stopped: bool,
        live_refresh_risk_acknowledged: bool = False,
    ) -> dict[str, Any]:
        registration_proof(request_id, proof)
        if (
            type(local_refreshers_stopped) is not bool
            or type(live_refresh_risk_acknowledged) is not bool
            or not (local_refreshers_stopped or live_refresh_risk_acknowledged)
        ):
            raise VisionError("invalid_request")
        body = {
            "version": 1,
            "handoff_secret": proof,
            "local_refreshers_stopped": local_refreshers_stopped,
        }
        if live_refresh_risk_acknowledged:
            body["live_refresh_risk_acknowledged"] = True
        value = self.client.request(
            "POST",
            f"/api/ai-accounts/registrations/{request_id}/confirm-handoff",
            body,
        )
        return registration_receipt(value, request_id)

    def cancel_registration(self, request_id: str, proof: str) -> dict[str, Any]:
        registration_proof(request_id, proof)
        value = self.client.request(
            "POST",
            f"/api/ai-accounts/registrations/{request_id}/cancel",
            {"version": 1, "handoff_secret": proof},
        )
        return registration_receipt(value, request_id)
