"""A synthetic upstream verifies request-time credentials and native streaming."""

import http.server
import json
import threading
import urllib.error
import urllib.request
from unittest.mock import Mock

import pytest

from claude_swap.exceptions import SessionError
from claude_swap.vision import VisionError
from claude_swap.vision_proxy import InferenceProxy


@pytest.fixture
def upstream():
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append((dict(self.headers), self.path, body))
            if b"redirect" in body:
                self.send_response(307)
                self.send_header("Location", "/stolen")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"data: synthetic\n\n")
            self.wfile.flush()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def request(proxy, path="/v1/messages?beta=true", *, token=None, body=b"{}"):
    return urllib.request.urlopen(
        urllib.request.Request(
            proxy.url + path,
            data=body,
            headers={
                "Authorization": "Bearer "
                + (proxy.capability if token is None else token),
                "Content-Type": "application/json",
                "Anthropic-Version": "2023-06-01",
                "X-Api-Key": "must-not-forward",
                "Cookie": "must-not-forward",
            },
        ),
        timeout=5,
    )


def test_each_request_gets_current_central_token_and_streams_upstream_response(
    upstream,
):
    url, seen = upstream
    credentials = Mock()
    credentials.get.side_effect = [
        {"accessToken": "synthetic-one"},
        {"accessToken": "synthetic-two"},
    ]
    with InferenceProxy(credentials, upstream=url) as proxy:
        for _ in range(2):
            with request(proxy) as response:
                assert response.read() == b"data: synthetic\n\n"
        assert credentials.get.call_count == 2
        env = proxy.environment({"CLAUDE_CODE_OAUTH_TOKEN": "old-token"})
        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == proxy.capability
        assert env["ANTHROPIC_BASE_URL"] == proxy.url
        assert [row[0]["Authorization"] for row in seen] == [
            "Bearer synthetic-one",
            "Bearer synthetic-two",
        ]
        assert all("X-Api-Key" not in row[0] and "Cookie" not in row[0] for row in seen)
        assert all(proxy.capability not in str(row) for row in seen)


@pytest.mark.parametrize(
    "path,token,status",
    [
        ("/v1/messages", "wrong", 401),
        ("/oauth/token", None, 404),
        ("/api/hosts", None, 404),
    ],
)
def test_auth_and_route_checks_happen_before_credential_issuance(
    upstream, path, token, status
):
    url, seen = upstream
    credentials = Mock()
    with InferenceProxy(credentials, upstream=url) as proxy:
        with pytest.raises(urllib.error.HTTPError) as error:
            request(proxy, path, token=token)
        assert error.value.code == status
    credentials.get.assert_not_called()
    assert not seen


def test_revoked_vision_access_stops_new_upstream_requests(upstream):
    url, seen = upstream
    credentials = Mock()
    credentials.get.side_effect = VisionError("not_permitted", 403)
    with InferenceProxy(credentials, upstream=url) as proxy:
        with pytest.raises(urllib.error.HTTPError) as error:
            request(proxy)
        assert error.value.code == 403
        assert json.loads(error.value.read())["type"] == "error"
    assert not seen


def test_upstream_redirect_is_not_followed_with_provider_credentials(upstream):
    url, seen = upstream
    credentials = Mock()
    credentials.get.return_value = {"accessToken": "synthetic-provider-token"}
    with InferenceProxy(credentials, upstream=url) as proxy:
        with pytest.raises(urllib.error.HTTPError) as error:
            request(proxy, body=b'{"redirect":true}')
        assert error.value.code == 502
    assert len(seen) == 1


def test_arbitrary_upstream_is_refused():
    with pytest.raises(SessionError, match="fixed provider"):
        InferenceProxy(Mock(), upstream="https://untrusted.example.invalid")


def test_native_runner_preserves_arguments_exit_status_and_hides_provider_token(
    monkeypatch,
):
    from types import SimpleNamespace

    from claude_swap.vision_proxy import run_native

    child = Mock()
    child.wait.return_value = 17
    child.poll.return_value = 17
    spawn = Mock(return_value=child)
    monkeypatch.setattr("claude_swap.vision_proxy.subprocess.Popen", spawn)
    launch = SimpleNamespace(
        env={
            "CLAUDE_CODE_OAUTH_TOKEN": "synthetic-provider-token",
            "CLAUDE_CONFIG_DIR": "/synthetic/profile",
        }
    )
    with pytest.raises(SystemExit) as exited:
        run_native(
            "/synthetic/claude", ["--resume", "conversation"], launch, Mock(), {}
        )
    assert exited.value.code == 17
    assert spawn.call_args.args == (["/synthetic/claude", "--resume", "conversation"],)
    env = spawn.call_args.kwargs["env"]
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] != "synthetic-provider-token"
    assert env["CLAUDE_CONFIG_DIR"] == "/synthetic/profile"
    assert env["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1:")
    child.terminate.assert_not_called()


def test_native_model_is_passed_to_central_selection(upstream):
    url, _ = upstream
    credentials = Mock()
    credentials.get.return_value = {"accessToken": "synthetic"}
    with (
        InferenceProxy(credentials, upstream=url) as proxy,
        request(proxy, body=b'{"model":"claude-sonnet-4-6"}') as response,
    ):
        response.read()
    credentials.get.assert_called_once_with(model="claude-sonnet-4-6")


@pytest.mark.parametrize("body", [b"null", b"[]", b'{"model":42}', b"not-json"])
def test_invalid_message_does_not_acquire_credentials(upstream, body):
    url, seen = upstream
    credentials = Mock()
    with InferenceProxy(credentials, upstream=url) as proxy:
        with pytest.raises(urllib.error.HTTPError) as error:
            request(proxy, body=body)
        assert error.value.code == 400
    credentials.get.assert_not_called()
    assert seen == []
