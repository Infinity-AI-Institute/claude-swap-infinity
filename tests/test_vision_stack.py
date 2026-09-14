"""Opt-in acceptance against an isolated real Vision server and local Supabase."""

import ipaddress
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

from claude_swap.vision import NoRedirects, configured_client
from claude_swap.vision_signin import VisionSignIn

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("VISION_TEST_URL"),
        reason="isolated Vision stack not supplied",
    ),
]


def local_origin(value):
    parsed = urllib.parse.urlsplit(value)
    assert parsed.scheme == "http" and parsed.netloc and parsed.path in ("", "/")
    assert (
        not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
    )
    host = parsed.hostname
    assert host == "localhost" or ipaddress.ip_address(host).is_loopback
    return value.rstrip("/")


def request(url, method="GET", body=None, headers=None):
    request_headers = dict(headers or {})
    payload = None
    if body is not None:
        payload = json.dumps(body).encode()
        request_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        url, data=payload, headers=request_headers, method=method
    )
    try:
        with urllib.request.build_opener(NoRedirects()).open(
            req, timeout=10
        ) as response:
            raw = response.read(2 * 1024 * 1024)
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        with error:
            raw = error.read(8192)
            return error.code, json.loads(raw) if raw else None


def test_browser_key_reaches_real_registry_but_not_hardware_and_revokes(
    tmp_path, monkeypatch
):
    web = local_origin(os.environ["VISION_TEST_URL"])
    supabase = local_origin(os.environ["SUPABASE_URL"])
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.setenv("VISION_API_URL", web)
    status, auth = request(
        supabase + "/auth/v1/token?grant_type=password",
        "POST",
        {"email": "luke@vision.dev", "password": "password123"},
        {"apikey": os.environ["SUPABASE_ANON_KEY"]},
    )
    assert status == 200, "Local seeded administrator sign-in failed"
    browser = {"Authorization": "Bearer " + auth["access_token"], "Origin": web}
    flow = VisionSignIn(tmp_path, web)
    public = flow.begin("claude-swap-stack-test")
    pending = flow.state.read("pending")
    key_id = None
    try:
        path = web + "/api/cli-auth/requests/" + public["request_id"]
        status, inspection = request(path, headers=browser)
        assert status == 200
        assert inspection["data"]["comparison_code"] == public["comparison_code"]
        status, _ = request(
            path + "/decision",
            "POST",
            {
                "comparison_code": public["comparison_code"],
                "approve": True,
            },
            browser,
        )
        assert status == 200
        for _ in range(8):
            result = flow.poll()
            if result["state"] == "signed_in":
                break
            time.sleep(result["retry_after_seconds"])
        else:
            pytest.fail("Approved request did not deliver a key")
        key_id = result["key_id"]
        client = configured_client(tmp_path)
        assert client is not None
        assert isinstance(client.discover(), list)
        status, _ = request(web + "/api/hosts", headers={"X-Api-Key": client.api_key})
        assert status == 403, "Registry-only key unexpectedly authorized hardware"
        status, _ = request(web + "/api/api-keys/" + key_id, "DELETE", headers=browser)
        assert status in (200, 204)
        status, _ = request(
            web + "/api/ai-accounts/registry?version=1&provider=claude&limit=100",
            headers={"X-Api-Key": client.api_key},
        )
        assert status == 401, "Revoked key remained usable"
    finally:
        # The saved proof also revokes a key if its issuance response was lost.
        request(
            web + "/api/cli-auth/requests/" + public["request_id"] + "/cancel",
            "POST",
            {"device_secret": pending["device_secret"]},
        )
        if key_id is not None:
            request(web + "/api/api-keys/" + key_id, "DELETE", headers=browser)
