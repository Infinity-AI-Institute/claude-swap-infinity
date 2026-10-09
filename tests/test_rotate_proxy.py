import http.server, json, threading, urllib.request

from claude_swap.rotate_proxy import LocalTokenPool, SharedRotateProxy


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
