import base64
import hashlib
import io
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from claude_swap.exceptions import SessionError
from claude_swap.vision import VisionError, VisionTransport
from claude_swap.vision_signin import VisionSignIn
from claude_swap.vision_state import VisionState

REQUEST = "00000000-0000-4000-8000-000000000001"
KEY_ID = "vkey_00000000-0000-4000-8000-000000000001"
SECRET = "vsk_" + "a" * 40


def fixture(tmp_path):
    store = SimpleNamespace(root=tmp_path)
    now = [2000000000.0]
    transport = VisionTransport("https://vision.example.invalid")
    response = {
        "request_id": REQUEST,
        "device_secret": "b" * 43,
        "comparison_code": "12345ABCDE",
        "verification_url": transport.url + "/cli-authorize/" + REQUEST,
        "expires_at": datetime.fromtimestamp(now[0] + 600, UTC).isoformat(),
        "interval_seconds": 5,
    }
    transport.request = Mock(return_value=response)
    flow = VisionSignIn(
        store.root, transport.url, transport=transport, now=lambda: now[0]
    )
    return store, now, transport, response, flow


def test_proofs_are_saved_before_public_url_and_resume_without_new_request(tmp_path):
    store, _now, transport, response, flow = fixture(tmp_path)
    public = flow.begin("test-host")
    pending = VisionState(store.root).read("pending")
    assert pending["device_secret"] == response["device_secret"]
    challenge = transport.request.call_args.args[2]["challenge"]
    assert challenge == base64.urlsafe_b64encode(
        hashlib.sha256(pending["verifier"].encode()).digest()
    ).decode().rstrip("=")
    assert pending["verifier"] not in json.dumps(public)
    assert pending["device_secret"] not in json.dumps(public)
    assert (
        VisionSignIn(store.root, transport.url, transport=transport).begin("test-host")
        == public
    )
    assert transport.request.call_count == 1


def test_lost_exchange_reply_preserves_proofs_and_backoff_across_restart(tmp_path):
    store, now, transport, _response, flow = fixture(tmp_path)
    flow.begin("test-host")
    assert (flow.poll())["retry_after_seconds"] == 5
    assert transport.request.call_count == 1
    now[0] += 5
    transport.request.side_effect = VisionError("slow_down", 429, 25)
    assert (flow.poll())["retry_after_seconds"] == 25
    first_proofs = transport.request.call_args.args[2].copy()
    transport.request.side_effect = None
    transport.request.return_value = {"api_key": SECRET, "key_id": KEY_ID}
    restarted = VisionSignIn(
        store.root, transport.url, transport=transport, now=lambda: now[0]
    )
    assert (restarted.poll())["retry_after_seconds"] == 25
    now[0] += 25
    result = restarted.poll()
    assert result == {"state": "signed_in", "url": transport.url, "key_id": KEY_ID}
    assert transport.request.call_args.args[2] == first_proofs
    assert SECRET not in json.dumps(result)
    assert VisionState(store.root).read("key")["api_key"] == SECRET
    assert VisionState(store.root).read("pending") is None


def test_failed_key_write_can_reconcile_the_original_issued_key(tmp_path, monkeypatch):
    store, now, transport, _response, flow = fixture(tmp_path)
    flow.begin("test-host")
    now[0] += 5
    transport.request.return_value = {"api_key": SECRET, "key_id": KEY_ID}
    original = flow.state.write

    def fail_key(name, value):
        if name == "key":
            raise OSError("simulated disk failure")
        original(name, value)

    monkeypatch.setattr(flow.state, "write", fail_key)
    with pytest.raises(OSError):
        flow.poll()
    assert VisionState(store.root).read("pending")["request_id"] == REQUEST
    now[0] += 5
    restarted = VisionSignIn(
        store.root, transport.url, transport=transport, now=lambda: now[0]
    )
    assert (restarted.poll())["key_id"] == KEY_ID


def test_cancel_keeps_recovery_state_on_network_failure(tmp_path):
    _store, _now, transport, _response, flow = fixture(tmp_path)
    flow.begin("test-host")
    transport.request.side_effect = VisionError("service_unavailable")
    with pytest.raises(VisionError):
        flow.cancel()
    assert flow.state.read("pending") is not None
    transport.request.side_effect = None
    transport.request.return_value = {"cancelled": True}
    assert flow.cancel() == {"state": "cancelled"}
    assert flow.state.read("pending") is None
    assert set(transport.request.call_args.args[2]) == {"device_secret"}


def test_verification_url_cannot_redirect_to_another_origin_or_include_proof(tmp_path):
    _store, _now, _transport, response, flow = fixture(tmp_path)
    response["verification_url"] = "https://different.example.invalid/?proof=private"
    with pytest.raises(VisionError) as result:
        flow.begin("test-host")
    assert "private" not in str(result.value)
    assert flow.state.read("pending") is None


def test_wrong_origin_does_not_receive_saved_proofs(tmp_path):
    store, _now, _transport, _response, flow = fixture(tmp_path)
    flow.begin("test-host")
    other = VisionTransport("https://different.example.invalid")
    other.request = Mock()
    with pytest.raises(SessionError, match="original origin"):
        VisionSignIn(store.root, other.url, transport=other).poll()
    other.request.assert_not_called()


def test_public_transport_omits_api_key_headers(monkeypatch):
    seen = []

    class Opener:
        def open(self, request, timeout):
            seen.append(request)
            return io.BytesIO(b'{"data":{"ok":true}}')

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    transport = VisionTransport("https://vision.example.invalid")
    assert transport.request("POST", "/api/cli-auth/requests", {}) == {"ok": True}
    assert "X-api-key" not in dict(seen[0].header_items())


def test_saved_key_is_used_after_restart_but_environment_takes_precedence(
    tmp_path, monkeypatch
):
    from claude_swap.vision import configured_client

    store, now, transport, _response, flow = fixture(tmp_path)
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.delenv("VISION_API_URL", raising=False)
    flow.begin("test-host")
    now[0] += 5
    transport.request.return_value = {"api_key": SECRET, "key_id": KEY_ID}
    flow.poll()
    client = configured_client(store.root)
    assert client.url == transport.url and client.api_key == SECRET
    monkeypatch.setenv("VISION_API_KEY", "environment-private-key")
    monkeypatch.setenv("VISION_API_URL", "https://different.example.invalid")
    assert configured_client(store.root).api_key == "environment-private-key"
    monkeypatch.delenv("VISION_API_KEY")
    with pytest.raises(SessionError, match="another origin"):
        configured_client(store.root)


def test_cancel_after_partial_local_completion_removes_the_revoked_key(
    tmp_path, monkeypatch
):
    _store, now, transport, _response, flow = fixture(tmp_path)
    flow.begin("test-host")
    now[0] += 5
    transport.request.return_value = {"api_key": SECRET, "key_id": KEY_ID}
    original = flow.state.remove

    def fail_cleanup(name):
        raise OSError("simulated interruption after key write")

    monkeypatch.setattr(flow.state, "remove", fail_cleanup)
    with pytest.raises(OSError):
        flow.poll()
    assert flow.state.read("key")["key_id"] == KEY_ID
    monkeypatch.setattr(flow.state, "remove", original)
    transport.request.return_value = {"cancelled": True}
    flow.cancel()
    assert flow.state.read("key") is None
    assert flow.state.read("pending") is None


def test_pending_request_cannot_be_relabelled_and_expired_proofs_are_not_sent(tmp_path):
    _store, now, transport, _response, flow = fixture(tmp_path)
    flow.begin("first-host")
    with pytest.raises(SessionError, match="another host label"):
        flow.begin("second-host")
    now[0] += 721
    with pytest.raises(VisionError) as result:
        flow.poll()
    assert result.value.code == "delivery_expired"
    assert transport.request.call_count == 1


def test_browser_cli_emits_only_public_approval_metadata(tmp_path, monkeypatch):
    from claude_swap import vision_cli

    _store, now, transport, response, flow = fixture(tmp_path)
    switcher = SimpleNamespace(backup_dir=tmp_path)
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.setattr(vision_cli, "VisionSignIn", lambda *args: flow)
    public = vision_cli.run_command(
        ["login", "--host-label", "test-host", "--no-wait"], switcher
    )
    assert public["comparison_code"] == response["comparison_code"]
    assert response["device_secret"] not in json.dumps(public)
    assert transport.request.call_args.args[2]["client"] == "claude-swap"
    now[0] += 5
    transport.request.return_value = {"api_key": SECRET, "key_id": KEY_ID}
    result = vision_cli.run_command(["status"], switcher)
    assert result["state"] == "signed_in"
    assert SECRET not in json.dumps(result)
    saved_status = vision_cli.run_command(["status"], switcher)
    assert saved_status == {"state": "signed_in", "url": transport.url}
    assert SECRET not in json.dumps(saved_status)


def test_environment_key_bypasses_browser_cli(tmp_path, monkeypatch):
    from claude_swap import vision_cli

    monkeypatch.setenv("VISION_API_KEY", "synthetic-env-key")
    factory = Mock(side_effect=AssertionError("browser initiation"))
    monkeypatch.setattr(vision_cli, "VisionSignIn", factory)
    result = vision_cli.run_command(["login"], SimpleNamespace(backup_dir=tmp_path))
    assert result["source"] == "environment"
    assert "synthetic-env-key" not in json.dumps(result)
