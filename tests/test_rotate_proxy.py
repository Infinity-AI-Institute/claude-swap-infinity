import http.server
import json
import threading
import urllib.error
import urllib.request
from unittest.mock import Mock

import pytest

from claude_swap.rotate_proxy import LocalTokenPool, SharedRotateProxy, control_command


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
