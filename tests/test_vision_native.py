"""Opt-in macOS native acceptance against a synthetic loopback provider only."""

import hashlib
import http.server
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

from claude_swap.session import SessionManager
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.vision import VisionClient
from claude_swap.vision_session import prepare_launch

NATIVE_HASH = "a506b6d970a4cf44f6abdb53a81ddcd5d3b0ce042a95c502fe9d1f946bdb8807"
TOKEN = "synthetic-vision-native-token"


@pytest.mark.skipif(
    sys.platform != "darwin" or not os.environ.get("CLAUDE_NATIVE_TEST_BINARY"),
    reason="Requires explicit pinned native binary and macOS sandbox-exec",
)
def test_prepared_access_only_launch_reaches_native_inference(temp_home, monkeypatch):
    native = Path(os.environ["CLAUDE_NATIVE_TEST_BINARY"]).resolve()
    assert hashlib.sha256(native.read_bytes()).hexdigest() == NATIVE_HASH
    requests = []
    message_counts = []

    class Provider(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 1_048_576:
                self.send_error(413)
                return
            body = json.loads(self.rfile.read(size))
            message_counts.append(len(body.get("messages", [])))
            expected_token = TOKEN if not requests else TOKEN + "-2"
            requests.append(
                {
                    "path": self.path.split("?")[0],
                    "correct_bearer": self.headers.get("Authorization")
                    == "Bearer " + expected_token,
                    "api_key": "x-api-key" in self.headers,
                    "stream": body.get("stream"),
                }
            )
            if requests[-1]["path"] != "/v1/messages" or len(requests) > 2:
                self.send_error(403)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            events = [
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_synthetic",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-sonnet-4-6",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "OK"},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 1},
                },
                {"type": "message_stop"},
            ]
            for event in events:
                self.wfile.write(
                    f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
                )
            self.wfile.flush()

    monkeypatch.delenv("VISION_API_KEY", raising=False)
    manager = SessionManager(ClaudeAccountSwitcher())
    registry = VisionClient("https://vision.example.invalid", "synthetic-registry-key")
    login = "ail_00000000-0000-4000-8000-000000000001"
    account = "aia_00000000-0000-4000-8000-000000000001"
    registry.credential = Mock(
        return_value={
            "email": "synthetic@example.invalid",
            "organization_id": "organization",
            "login_id": login,
            "account_id": account,
            "generation": 1,
            "expires_at": None,
            "accessToken": TOKEN,
        }
    )
    record = {
        "visionUrl": registry.url,
        "visionLoginId": login,
        "visionAccountId": account,
        "email": "synthetic@example.invalid",
        "organizationUuid": "organization",
    }
    launch = prepare_launch(manager, record, registry, share=False, share_history=False)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    # Only the synthetic endpoint is reachable. Helpers, Keychain access, and
    # reads of user homes are denied by the kernel, independently of CLI flags.
    sandbox = f'''(version 1)
(allow default)
(deny network*)
(allow network-outbound (remote ip "localhost:{port}"))
(deny process-exec)
(allow process-exec (literal "{native}"))
(deny file-read* (subpath "/Users"))
(allow file-read* (literal "{native}"))
'''
    env = {
        key: launch.env[key]
        for key in (
            "CLAUDE_CONFIG_DIR",
            "CLAUDE_SECURESTORAGE_CONFIG_DIR",
            "CLAUDE_CODE_OAUTH_TOKEN",
        )
    }
    env.update(
        {
            "PATH": "/usr/bin:/bin",
            "HOME": str(temp_home),
            "TMPDIR": str(temp_home),
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
        }
    )
    command = [
        "/usr/bin/sandbox-exec",
        "-p",
        sandbox,
        str(native),
        "-p",
        "Reply with OK.",
        "--safe-mode",
        "--tools",
        "",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--setting-sources",
        "",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        "claude-sonnet-4-6",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=temp_home,
            env=env,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, "Pinned native synthetic inference failed"
        first_events = [
            json.loads(line)
            for line in result.stdout.splitlines()
            if line.startswith(b"{")
        ]
        session_id = next(
            event["session_id"]
            for event in first_events
            if event.get("type") == "result"
        )
        registry.credential.return_value = {
            **registry.credential.return_value,
            "accessToken": TOKEN + "-2",
            "generation": 2,
        }
        resumed = prepare_launch(
            manager, record, registry, share=False, share_history=False
        )
        assert resumed.directory == launch.directory
        env["CLAUDE_CODE_OAUTH_TOKEN"] = resumed.env["CLAUDE_CODE_OAUTH_TOKEN"]
        result = subprocess.run(
            command + ["--resume", session_id],
            cwd=temp_home,
            env=env,
            capture_output=True,
            timeout=30,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
    assert result.returncode == 0, "Pinned native synthetic inference failed"
    assert (
        requests
        == [
            {
                "path": "/v1/messages",
                "correct_bearer": True,
                "api_key": False,
                "stream": True,
            }
        ]
        * 2
    )
    events = [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith(b"{")
    ]
    assert any(
        event.get("type") == "result" and not event.get("is_error") for event in events
    )
    assert message_counts[1] > message_counts[0]
    assert any(
        event.get("type") == "result" and event.get("session_id") == session_id
        for event in events
    )
    assert not (launch.directory / ".credentials.json").exists()
