"""Process-local native inference adapter with request-time central credentials.

Only the native message API is forwarded. The per-process capability is not a
provider credential; real access tokens remain in the parent and are attached to
the fixed Anthropic endpoint. Refresh tokens are never accepted or stored here.
"""

from __future__ import annotations

import http.server
import json
import secrets
import signal
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request

from claude_swap.exceptions import ClaudeSwitchError, SessionError
from claude_swap.vision import VisionError
from claude_swap.vision_session import acquire_credential, recover_rejected_credential

MAX_REQUEST_BYTES = 64 * 1024 * 1024
UPSTREAM = "https://api.anthropic.com"
FORWARDED_HEADERS = {
    "anthropic-version",
    "anthropic-beta",
    "content-type",
    "accept",
    "user-agent",
    "x-app",
}
RESPONSE_HEADERS = {
    "content-type",
    "content-encoding",
    "content-length",
    "request-id",
    "retry-after",
}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class CentralCredential:
    def __init__(self, client, record):
        self.client = client
        self.record = dict(record)
        self.lock = threading.Lock()

    def get(self, *, model=None):
        with self.lock:
            credential = acquire_credential(self.client, self.record)
            if (
                credential["email"] != self.record["email"]
                or credential["organization_id"] != self.record["organizationUuid"]
            ):
                raise SessionError(
                    "The central login identity changed; synchronize accounts before continuing."
                )
            self.record["visionGeneration"] = credential["generation"]
            return credential

    def recover(self, rejected, *, model=None):
        with self.lock:
            return recover_rejected_credential(self.client, rejected)

    def record_rate_limit(self, credential, retry_after):
        pass

    def rate_limited(self, credential, retry_after, *, model=None):
        return None


class InferenceProxy:
    def __init__(self, credentials, *, upstream=UPSTREAM):
        parsed = urllib.parse.urlsplit(upstream)
        # Loopback is supported for synthetic acceptance; callers cannot choose
        # arbitrary hosts or follow a provider redirect with an access token.
        if upstream != UPSTREAM and not (
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost"}
            and parsed.port is not None
            and not parsed.username
            and not parsed.password
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
        ):
            raise SessionError(
                "The native inference upstream must be the fixed provider endpoint."
            )
        self.credentials = credentials
        self.upstream = upstream.rstrip("/")
        self.capability = secrets.token_urlsafe(32)
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )
        self.server = None
        self.thread = None

    @property
    def url(self):
        if self.server is None:
            raise SessionError("The native inference adapter is not running.")
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        proxy = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def _error(self, status, message, retry_after=None):
                body = json.dumps(
                    {
                        "type": "error",
                        "error": {"type": "api_error", "message": message},
                    }
                ).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                if retry_after is not None:
                    self.send_header("Retry-After", str(retry_after))
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                self.wfile.write(body)

            def do_POST(self):
                self.connection.settimeout(30)
                self.close_connection = True
                expected = "Bearer " + proxy.capability
                provided = self.headers.get("Authorization", "")
                if not secrets.compare_digest(
                    provided.encode(), expected.encode()
                ) or self.headers.get("Origin"):
                    self._error(401, "Native adapter authorization required.")
                    return
                parsed = urllib.parse.urlsplit(self.path)
                if (
                    parsed.scheme
                    or parsed.netloc
                    or parsed.path not in {"/v1/messages", "/v1/messages/count_tokens"}
                ):
                    self._error(404, "This native API route is not supported.")
                    return
                lengths = self.headers.get_all("Content-Length", [])
                if (
                    len(lengths) != 1
                    or not lengths[0].isascii()
                    or not lengths[0].isdecimal()
                    or self.headers.get("Transfer-Encoding")
                ):
                    self._error(400, "A bounded native request body is required.")
                    return
                length = int(lengths[0])
                if not 0 < length <= MAX_REQUEST_BYTES:
                    self._error(413, "The native request body is too large.")
                    return
                body = self.rfile.read(length)
                if len(body) != length:
                    self._error(400, "The native request body is incomplete.")
                    return
                try:
                    payload = json.loads(body)
                    if not isinstance(payload, dict):
                        raise TypeError()
                    model = payload.get("model")
                    if model is not None and not isinstance(model, str):
                        raise TypeError()
                except (ValueError, TypeError, UnicodeDecodeError):
                    self._error(400, "A native JSON message object is required.")
                    return
                try:
                    credential = proxy.credentials.get(model=model)
                except VisionError as error:
                    status = error.status if error.status in {401, 403, 429} else 503
                    self._error(
                        status,
                        "Central credentials are unavailable; " + error.code + ".",
                        error.retry_after_seconds,
                    )
                    return
                except ClaudeSwitchError:
                    self._error(
                        503,
                        "Central account configuration or recovery needs attention.",
                    )
                    return
                headers = {
                    key: value
                    for key, value in self.headers.items()
                    if key.lower() in FORWARDED_HEADERS
                    or key.lower().startswith("x-stainless-")
                }
                response_started = False
                try:
                    response = proxy.open_message(self.path, body, headers, credential)
                    if response.status == 401:
                        # An explicit authentication rejection precedes inference.
                        # Close it before central recovery, and replay at most once.
                        response.close()
                        replacement = proxy.credentials.recover(credential, model=model)
                        if replacement is None:
                            self._error(
                                401, "Central login reauthentication is required."
                            )
                            return
                        credential = replacement
                        response = proxy.open_message(
                            self.path, body, headers, credential
                        )
                    elif response.status == 429:
                        try:
                            replacement = proxy.credentials.rate_limited(
                                credential,
                                response.headers.get("Retry-After"),
                                model=model,
                            )
                        except BaseException:
                            response.close()
                            raise
                        if replacement is not None:
                            response.close()
                            credential = replacement
                            response = proxy.open_message(
                                self.path, body, headers, credential
                            )
                    with response:
                        if response.status == 429:
                            proxy.credentials.record_rate_limit(
                                credential, response.headers.get("Retry-After")
                            )
                        status = response.status
                        if 300 <= status < 400:
                            self._error(
                                502, "The provider returned an unsupported redirect."
                            )
                            return
                        response_started = True
                        self.send_response(status)
                        for key, value in response.headers.items():
                            if (
                                key.lower() in RESPONSE_HEADERS
                                or key.lower().startswith("anthropic-ratelimit-")
                            ):
                                self.send_header(key, value)
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        while chunk := response.read1(65536):
                            self.wfile.write(chunk)
                            self.wfile.flush()
                except VisionError as error:
                    status = error.status if error.status in {401, 403, 429} else 503
                    self._error(
                        status,
                        "Central credential recovery is unavailable.",
                        error.retry_after_seconds,
                    )
                except ClaudeSwitchError:
                    self._error(503, "No central login is available for recovery.")
                except (urllib.error.URLError, TimeoutError):
                    # No retry here: only the native client decides whether to
                    # replay a request whose upstream execution is uncertain.
                    if not response_started:
                        self._error(502, "The provider connection is unavailable.")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def open_message(self, path, body, headers, credential):
        headers = {**headers, "Authorization": "Bearer " + credential["accessToken"]}
        request = urllib.request.Request(
            self.upstream + path, data=body, headers=headers, method="POST"
        )
        try:
            return self.opener.open(request, timeout=120)
        except urllib.error.HTTPError as error:
            return error

    def __exit__(self, *_args):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def environment(self, native_env):
        env = dict(native_env)
        env["ANTHROPIC_BASE_URL"] = self.url
        env["CLAUDE_CODE_OAUTH_TOKEN"] = self.capability
        return env


def run_native(native, arguments, launch, client, record, *, switcher=None):
    """Keep the adapter alive while native owns terminal input and output."""
    if switcher is None:
        credentials = CentralCredential(client, record)
    else:
        from claude_swap.vision_pool import CentralPoolCredential

        credentials = CentralPoolCredential(switcher, client, record)
    with InferenceProxy(credentials) as proxy:
        child = subprocess.Popen(
            [native, *arguments], env=proxy.environment(launch.env)
        )
        previous = {}
        try:
            if threading.current_thread() is threading.main_thread():
                # Terminal SIGINT reaches both processes. Let native interpret
                # it (cancel a turn or exit); a Python KeyboardInterrupt must not
                # tear down the adapter while native is still using it.
                previous[signal.SIGINT] = signal.signal(signal.SIGINT, lambda *_: None)

                def terminate(_signum, _frame):
                    if child.poll() is None:
                        child.terminate()

                previous[signal.SIGTERM] = signal.signal(signal.SIGTERM, terminate)
            returncode = child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    raise SystemExit(returncode)
