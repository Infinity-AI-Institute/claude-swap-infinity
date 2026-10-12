import http.server
import json
import os
import stat
import sys
import threading
import types
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import Mock

import pytest

from claude_swap import cli, rotate_proxy
from claude_swap.exceptions import ClaudeSwitchError, NoUsableLogin
from claude_swap.models import Platform
from claude_swap.rotate_proxy import LocalTokenPool, SharedRotateProxy, control_command
from claude_swap.switcher import ClaudeAccountSwitcher


@pytest.fixture(autouse=True)
def quick_server_shutdown(monkeypatch):
    serve = http.server.ThreadingHTTPServer.serve_forever
    monkeypatch.setattr(http.server.ThreadingHTTPServer, "serve_forever",
                        lambda self, poll_interval=0.01: serve(self, poll_interval))


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


def accounts():
    return [{"num": "3", "name": "a", "token": "sk-ant-oat01-a"},
            {"num": "4", "name": "b", "token": "sk-ant-oat01-b"}]


def test_sticky_then_rotate_then_return():
    clock = Clock()
    pool = LocalTokenPool(accounts(), clock=clock)
    first = pool.get()
    assert first["name"] == "a" and pool.get()["name"] == "a"
    nxt = pool.rate_limited(first, "60")
    assert nxt["name"] == "b" and pool.get()["name"] == "b"
    clock.t = 61
    assert pool.get()["name"] == "b"  # sticky: no switch back until b is limited
    assert pool.rate_limited(pool.get(), None)["name"] == "a"


def test_all_cooling_returns_none_and_401_drops():
    pool = LocalTokenPool(accounts(), clock=Clock())
    a = pool.get(); b = pool.rate_limited(a, "60")
    assert pool.rate_limited(b, "60") is None
    pool2 = LocalTokenPool(accounts(), clock=Clock())
    assert pool2.recover(pool2.get())["name"] == "b"
    assert [s["dead"] for s in pool2.status()] == [True, False]


def test_proxy_replays_429_on_next_account():
    seen = []

    class Upstream(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            auth = self.headers["Authorization"]; seen.append(auth)
            status = 429 if auth.endswith("-a") else 200
            body = json.dumps({"ok": status == 200}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if status == 429: self.send_header("Retry-After", "120")
            self.end_headers(); self.wfile.write(body)

    up = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    pool = LocalTokenPool(accounts(), clock=Clock())
    with SharedRotateProxy(pool, port=0, capability="cap",
                           upstream=f"http://127.0.0.1:{up.server_port}") as proxy:
        for _ in range(2):
            req = urllib.request.Request(proxy.url + "/v1/messages", method="POST",
                data=json.dumps({"model": "m"}).encode(),
                headers={"Authorization": "Bearer cap", "Content-Type": "application/json"})
            assert json.load(urllib.request.urlopen(req)) == {"ok": True}
    up.shutdown()
    assert seen == ["Bearer sk-ant-oat01-a", "Bearer sk-ant-oat01-b", "Bearer sk-ant-oat01-b"]


def three_accounts():
    return accounts() + [{"num": "5", "name": "c", "token": "sk-ant-oat01-c"}]


@pytest.fixture
def rotating_upstream():
    seen = []
    statuses = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            account = self.headers["Authorization"].rsplit("-", 1)[-1]
            seen.append((account, self.path, body))
            self.send_response(statuses.get(account, 200))
            self.send_header("Retry-After", "120")
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"data: ok\n\n")

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", statuses, seen
    finally:
        server.shutdown()
        server.server_close()


def post(proxy, path="/v1/messages", payload=None, capability="cap", **headers):
    return urllib.request.urlopen(urllib.request.Request(
        proxy.url + path,
        data=json.dumps(payload if payload is not None else {"model": "m", "stream": True}).encode(),
        headers={"Authorization": "Bearer " + capability, "Content-Type": "application/json", **headers},
    ), timeout=3)


@pytest.mark.parametrize("statuses,expected", [
    ({"a": 429, "b": 429}, 200),
    ({"a": 401, "b": 429}, 200),
    ({"a": 429, "b": 401}, 200),
    ({"a": 429, "b": 429, "c": 429}, 429),
    ({"a": 500}, 500),
])
def test_pool_retries_all_accounts_and_preserves_stream(rotating_upstream, statuses, expected):
    url, replies, seen = rotating_upstream
    replies.update(statuses)
    pool = LocalTokenPool(three_accounts(), clock=Clock())
    with SharedRotateProxy(pool, port=0, capability="cap", upstream=url) as proxy:
        if expected == 200:
            with post(proxy) as response:
                assert response.read() == b"data: ok\n\n"
        else:
            with pytest.raises(urllib.error.HTTPError) as error:
                post(proxy)
            assert error.value.code == expected
            assert error.value.headers["Retry-After"] == "120"
            error.value.close()
    assert [a for a, _, _ in seen] == (["a"] if expected == 500 else ["a", "b", "c"])
    assert len({body for _, _, body in seen}) == 1
    if expected == 429:
        assert all(row["cooling_s"] == 120 for row in pool.status())


def test_reset_via_cli_reuses_account_without_restarting(rotating_upstream, tmp_path, capsys):
    url, replies, seen = rotating_upstream
    pool = LocalTokenPool(three_accounts(), clock=Clock())
    for account in pool.accounts:
        pool.record_rate_limit(pool._credential(account), "120")
    switcher = Mock(backup_dir=tmp_path)
    (tmp_path / "rotate-capability").write_text("cap")
    with SharedRotateProxy(pool, port=0, capability="cap", upstream=url) as proxy:
        port = proxy.server.server_port
        control_command(switcher, "rotate-status", [], port=port)
        assert all(a["cooling_s"] == 120 for a in json.loads(capsys.readouterr().out)["accounts"])
        control_command(switcher, "rotate-reset", ["b"], port=port)
        result = json.loads(capsys.readouterr().out)
        assert result["reset"] == "b"
        assert [a["cooling_s"] for a in result["accounts"]] == [120, 0, 120]
        assert result["accounts"][1]["active"]
        with post(proxy) as response:
            assert response.status == 200
        assert seen[0][0] == "b"
        with post(proxy, "/local/rotation/reset", {"account": "5"}) as response:
            assert json.load(response)["reset"] == "c"


@pytest.mark.parametrize("capability,headers,payload,expected", [
    ("wrong", {}, {"account": "b"}, 401),
    ("cap", {"Origin": "https://example.invalid"}, {"account": "b"}, 401),
    ("cap", {}, {"account": "missing"}, 400),
    ("cap", {}, {"account": []}, 400),
])
def test_control_rejects_invalid_requests(rotating_upstream, capability, headers, payload, expected):
    url, _, seen = rotating_upstream
    pool = LocalTokenPool(three_accounts(), clock=Clock())
    pool.record_rate_limit(pool.get(), "120")
    before = pool.status()
    with SharedRotateProxy(pool, port=0, capability="cap", upstream=url) as proxy:
        with pytest.raises(urllib.error.HTTPError) as error:
            post(proxy, "/local/rotation/reset", payload, capability, **headers)
        assert error.value.code == expected
        error.value.close()
    assert pool.status() == before
    assert seen == []


def test_reset_does_not_revive_rejected_token():
    pool = LocalTokenPool(accounts(), clock=Clock())
    pool.recover(pool.get())
    with pytest.raises(ValueError, match="rejected"):
        pool.reset("a")
    assert pool.status()[0]["dead"]


def test_retry_never_replays_same_token(rotating_upstream, monkeypatch):
    url, replies, seen = rotating_upstream
    replies["a"] = 429
    pool = LocalTokenPool(three_accounts(), clock=Clock())
    first = pool.get()
    monkeypatch.setattr(pool, "rate_limited", lambda *a, **kw: first)
    with SharedRotateProxy(pool, port=0, capability="cap", upstream=url) as proxy:
        with pytest.raises(urllib.error.HTTPError) as error:
            post(proxy)
        assert error.value.code == 429
        error.value.close()
    assert [row[0] for row in seen] == ["a"]


def test_single_account_exhaustion_is_recorded(rotating_upstream):
    url, replies, seen = rotating_upstream
    replies["a"] = 429
    pool = LocalTokenPool(accounts()[:1], clock=Clock())
    with SharedRotateProxy(pool, port=0, capability="cap", upstream=url) as proxy:
        with pytest.raises(urllib.error.HTTPError) as error:
            post(proxy)
        assert error.value.code == 429
        error.value.close()
    assert len(seen) == 1
    assert pool.status()[0]["cooling_s"] == 120


# ---------------------------------------------------------------------------
# Which stored accounts are pooled
# ---------------------------------------------------------------------------

API_KEY = "sk-ant-api03-" + "a1b2c3d4e5" * 4


def file_backed_switcher() -> ClaudeAccountSwitcher:
    """A real switcher on the isolated test home, keeping credentials in files."""
    switcher = ClaudeAccountSwitcher()
    switcher.platform = Platform.LINUX
    switcher._setup_directories()
    switcher._init_sequence_file()
    return switcher


def test_pool_holds_every_enabled_setup_token_and_nothing_else(temp_home: Path):
    switcher = file_backed_switcher()
    switcher.add_account_from_token("sk-ant-oat01-default-label")
    # `add-token --email` stores a setup token under a real address.
    switcher.add_account_from_token("sk-ant-oat01-custom-email", email="me@example.com")
    # An API key bills per token and must never be pooled.
    switcher.add_account_from_token(API_KEY)
    # A browser login has a refresh token that Claude Code owns and rotates.
    switcher.add_account_from_token("sk-ant-oat01-login", email="login@example.com")
    switcher._write_account_credentials("4", "login@example.com", json.dumps({
        "claudeAiOauth": {
            "accessToken": "sk-ant-oat01-login",
            "refreshToken": "sk-ant-ort01-refresh",
            "expiresAt": 9,
        }
    }))
    switcher.add_account_from_token("sk-ant-oat01-disabled")
    switcher.set_account_disabled("5", True)

    pooled = rotate_proxy.load_token_accounts(switcher)

    assert [(account["num"], account["token"]) for account in pooled] == [
        ("1", "sk-ant-oat01-default-label"),
        ("2", "sk-ant-oat01-custom-email"),
    ]


def test_machine_without_an_account_store_has_nothing_to_pool(temp_home: Path):
    assert rotate_proxy.load_token_accounts(ClaudeAccountSwitcher()) == []


# ---------------------------------------------------------------------------
# Cooldowns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("retry_after,cooldown_seconds", [
    ("120", 120),
    ("5", 30),  # never retry sooner than 30 seconds
    ("999999", 6 * 3600),  # never set an account aside for longer than 6 hours
    (None, 15 * 60),
    ("Wed, 21 Oct 2026 07:28:00 GMT", 15 * 60),  # only the seconds form is read
])
def test_cooldown_follows_retry_after_within_bounds(retry_after, cooldown_seconds):
    pool = LocalTokenPool(accounts(), clock=Clock())
    pool.record_rate_limit(pool.get(), retry_after)
    assert pool.status()[0]["cooling_s"] == cooldown_seconds


# ---------------------------------------------------------------------------
# No account can serve
# ---------------------------------------------------------------------------


def test_pool_with_every_token_rejected_reports_no_usable_login():
    pool = LocalTokenPool(accounts(), clock=Clock())
    second = pool.recover(pool.get())
    with pytest.raises(NoUsableLogin, match="rejected"):
        pool.recover(second)
    with pytest.raises(NoUsableLogin, match="rejected"):
        pool.get()


def test_rejection_while_the_rest_are_rate_limited_reports_when_one_returns():
    clock = Clock()
    pool = LocalTokenPool(accounts(), clock=clock)
    second = pool.rate_limited(pool.get(), "120")
    clock.t = 20
    with pytest.raises(NoUsableLogin, match="rate-limited") as error:
        pool.recover(second)
    assert error.value.retry_after_seconds == 100
    assert [row["dead"] for row in pool.status()] == [False, True]


@pytest.mark.xfail(strict=True, reason="the full-pool retry stops before reporting a "
                   "rejection on its last attempt; fixed in the next commit")
def test_proxy_tells_the_agent_not_to_retry_when_no_token_can_serve(rotating_upstream):
    url, statuses, seen = rotating_upstream
    statuses["a"] = 401
    pool = LocalTokenPool(accounts()[:1], clock=Clock())
    with SharedRotateProxy(pool, port=0, capability="cap", upstream=url) as proxy:
        with pytest.raises(urllib.error.HTTPError) as error:
            post(proxy)
        message = json.loads(error.value.read())["error"]["message"]
        error.value.close()
    assert error.value.code == 503
    # Without this header Claude retries a 503 for minutes before showing it.
    assert error.value.headers["x-should-retry"] == "false"
    assert "rejected" in message and "add-token" in message
    # `cswap run` advice belongs to the per-session adapter, not this pool.
    assert "cswap run" not in message
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# Who may call the proxy, and where it listens
# ---------------------------------------------------------------------------


def test_caller_without_the_capability_never_reaches_the_provider(rotating_upstream):
    url, _statuses, seen = rotating_upstream
    pool = LocalTokenPool(accounts(), clock=Clock())
    with SharedRotateProxy(pool, port=0, capability="cap", upstream=url) as proxy:
        with pytest.raises(urllib.error.HTTPError) as error:
            post(proxy, capability="wrong")
        assert error.value.code == 401
        error.value.close()
    assert seen == []


def test_occupied_port_is_a_usage_error(rotating_upstream):
    url, _statuses, _seen = rotating_upstream
    with SharedRotateProxy(
        LocalTokenPool(accounts()), port=0, capability="cap", upstream=url
    ) as running:
        second = SharedRotateProxy(
            LocalTokenPool(accounts()),
            port=running.server.server_port,
            capability="cap",
            upstream=url,
        )
        with pytest.raises(ClaudeSwitchError, match="CSWAP_ROTATE_PORT"):
            second.__enter__()


def store_in(directory: Path) -> types.SimpleNamespace:
    """The one switcher attribute the capability file depends on."""
    return types.SimpleNamespace(backup_dir=directory)


def test_capability_is_created_private_and_then_reused(tmp_path: Path):
    capability = rotate_proxy.load_or_create_capability(store_in(tmp_path))
    assert len(capability) >= 32
    assert rotate_proxy.load_or_create_capability(store_in(tmp_path)) == capability
    if sys.platform != "win32":
        mode = stat.S_IMODE(os.stat(tmp_path / "rotate-capability").st_mode)
        assert mode == 0o600


def test_capability_waits_for_a_concurrent_creator_to_finish_writing(tmp_path: Path):
    capability_file = tmp_path / "rotate-capability"
    capability_file.touch()  # another process created the file and has not written it yet
    writer = threading.Timer(0.05, capability_file.write_text, args=("from-the-other-process",))
    writer.start()
    try:
        capability = rotate_proxy.load_or_create_capability(store_in(tmp_path))
    finally:
        writer.join()
    assert capability == "from-the-other-process"


def test_empty_capability_file_is_an_error_not_an_empty_capability(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rotate_proxy, "CAPABILITY_WRITE_WAIT_S", 0.05)
    (tmp_path / "rotate-capability").touch()
    with pytest.raises(ClaudeSwitchError, match="empty"):
        rotate_proxy.load_or_create_capability(store_in(tmp_path))


def test_capability_needs_an_account_store(tmp_path: Path):
    with pytest.raises(ClaudeSwitchError, match="add-token"):
        rotate_proxy.load_or_create_capability(store_in(tmp_path / "missing"))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def test_serve_announces_its_address_and_accounts_until_interrupted(temp_home: Path, monkeypatch):
    switcher = file_backed_switcher()
    switcher.add_account_from_token("sk-ant-oat01-one")
    announcements = []

    def interrupt(_seconds):
        raise KeyboardInterrupt

    # serve() only sleeps while it waits to be interrupted.
    monkeypatch.setattr(rotate_proxy.time, "sleep", interrupt)
    rotate_proxy.serve(switcher, port=0, log=lambda message, **_: announcements.append(message))

    assert len(announcements) == 1
    assert announcements[0].startswith("rotate-proxy on http://127.0.0.1:")
    assert announcements[0].endswith("with 1 accounts: setup-token-1@token.local")


def run_cli(monkeypatch, *arguments: str) -> None:
    monkeypatch.setattr(sys, "argv", ["cswap", *arguments])
    cli.main()


def test_rotate_env_prints_the_endpoint_and_capability(temp_home: Path, monkeypatch, capsys):
    switcher = file_backed_switcher()
    monkeypatch.setenv("CSWAP_ROTATE_PORT", "45000")
    run_cli(monkeypatch, "rotate-env")
    capability = rotate_proxy.load_or_create_capability(switcher)
    assert capsys.readouterr().out.splitlines() == [
        "ANTHROPIC_BASE_URL=http://127.0.0.1:45000",
        f"CLAUDE_CODE_OAUTH_TOKEN={capability}",
    ]


@pytest.mark.parametrize("port", ["abc", "0", "70000", ""])
def test_invalid_port_is_a_usage_error(temp_home: Path, monkeypatch, capsys, port):
    file_backed_switcher()
    monkeypatch.setenv("CSWAP_ROTATE_PORT", port)
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, "rotate-env")
    assert exit_info.value.code == 1
    captured = capsys.readouterr()
    assert "CSWAP_ROTATE_PORT" in captured.err
    assert captured.out == ""


def test_rotate_proxy_without_setup_tokens_is_a_usage_error(temp_home: Path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, "rotate-proxy")
    assert exit_info.value.code == 1
    assert "add-token" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["rotate-proxy", "rotate-env"])
def test_rotation_commands_explain_themselves(monkeypatch, capsys, command):
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, command, "--help")
    assert exit_info.value.code == 0
    assert "CSWAP_ROTATE_PORT" in capsys.readouterr().out


def test_main_help_lists_the_rotation_commands(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, "--help")
    help_text = capsys.readouterr().out
    assert "rotate-proxy" in help_text and "rotate-env" in help_text
