"""Process-local native inference adapter with request-time central credentials.

Only the native message API is forwarded. The per-process capability is not a
provider credential; real access tokens remain in the parent and are attached to
the fixed Anthropic endpoint. Refresh tokens are never accepted or stored here.
"""

from __future__ import annotations

import http.client
import http.server
import json
import logging
import secrets
import signal
import socket
import struct
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request

from claude_swap.exceptions import ClaudeSwitchError, NoUsableLogin, SessionError
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
# Appended to the pool's explanation when no Vision login can serve a request.
# A launch that names no account falls back to a local login, so relaunching
# that way is the way out.
NO_USABLE_LOGIN_ADVICE = (
    "Exit and rerun `cswap run` without naming an account to fall back to this "
    "machine's own Claude login if it has one, or wait for the reset."
)

# How native Claude (2.1.311, measured against a fake upstream) shows an
# adapter error: it puts "Please run /login" before every 401 and 403, which
# can never fix a Vision-side problem, so the adapter sends neither. A 400 is
# shown verbatim and never retried: the status for problems a person must fix.
# A 5xx with ``x-should-retry: false`` is shown at once, with "This is a
# server-side issue, usually temporary" appended; without that header, native
# retries a 5xx or 429 for about three minutes before showing it.
NEEDS_ACTION_STATUS = 400

# Sent when native presents a login other than this adapter's capability,
# which happens after /login inside a swap session.
STRAY_LOGIN_MESSAGE = (
    "cswap: this Claude session is using a login other than the one cswap "
    "gave it, usually because /login was run inside it. /login cannot fix a "
    "cswap session: exit and relaunch, and the launch moves that login aside."
)


def rejected_key_message(key_source: str | None) -> str:
    """What to do when Vision refuses the API key, even after re-reading it."""
    message = (
        "cswap: Vision rejected this session's Vision API key (revoked or "
        "unknown). Create a new key at https://vision.infinity.inc (gear → "
        "API keys) and save it with `cswap --set-vision-token`, which writes "
        "~/.config/vision/credentials.json; "
    )
    if key_source == "environment":
        return message + (
            "this session read its key from VISION_API_KEY, which takes "
            "precedence, so also update VISION_API_KEY and relaunch."
        )
    return message + "this session picks it up on your next message."

# The adapter runs in the process whose terminal native Claude draws on, so it
# must never write to stderr. Without a handler of its own, logging prints
# warnings there as a last resort; with one, they reach only cswap's log file.
_log = logging.getLogger("claude-swap.vision_proxy")
_log.addHandler(logging.NullHandler())


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _AdapterServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # socketserver prints the traceback to stderr, over native's screen.
        _log.error("Native adapter request failed unexpectedly.", exc_info=True)


class CentralCredential:
    def __init__(self, client, record, *, wait_seconds=90):
        self.client = client
        self.record = dict(record)
        self.wait_seconds = wait_seconds
        self.lock = threading.Lock()

    def get(self, *, model=None):
        with self.lock:
            credential = acquire_credential(
                self.client, self.record, wait_seconds=self.wait_seconds
            )
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

    def relogin_command(self, credential):
        from claude_swap.vision_pool import relogin_command

        return relogin_command(self.record)


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
        self._announce_relogin()
        proxy = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def _error(self, status, message, retry_after=None, *, retryable=True):
                error_type = (
                    "invalid_request_error"
                    if status == NEEDS_ACTION_STATUS
                    else "api_error"
                )
                body = json.dumps(
                    {
                        "type": "error",
                        "error": {"type": error_type, "message": message},
                    }
                ).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                if retry_after is not None:
                    self.send_header("Retry-After", str(retry_after))
                if not retryable:
                    # The Anthropic SDK and native Claude honor this header.
                    # Without it, native retries a 503 ten times (about three
                    # minutes in Claude 2.1.283) before showing the message.
                    self.send_header("x-should-retry", "false")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                self.wfile.write(body)

            def _reset(self):
                """Drop native's connection with a reset, not a clean close.

                A clean close would end an event stream as though the reply
                were complete; a reset makes native see the failure and retry.
                """
                self.close_connection = True
                try:
                    self.connection.setsockopt(
                        socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                    )
                    # Closed now, before socketserver's shutdown would send a FIN.
                    self.connection.close()
                except OSError:
                    pass

            def _needs_action(self, message):
                """Fail native's turn at once with what a person must do."""
                self._error(NEEDS_ACTION_STATUS, message, retryable=False)

            def _no_usable_login(self, error):
                """Fail native's turn with why no login can serve, naming each."""
                from claude_swap.vision_pool import ProviderRateLimited

                _log.warning("No Vision login can serve a request: %s", error)
                if isinstance(error, ProviderRateLimited):
                    # Every login is inside a provider Retry-After window, so
                    # native's own retry after it is expected to succeed.
                    self._error(429, f"cswap: {error}", error.retry_after_seconds)
                elif getattr(error, "needs_action", False):
                    self._needs_action(f"cswap: {error}")
                else:
                    self._error(
                        503,
                        f"cswap: {error} {NO_USABLE_LOGIN_ADVICE}",
                        error.retry_after_seconds,
                        retryable=False,
                    )

            def _vision_failure(self, error):
                """Answer a Vision refusal or outage that affects every login."""
                _log.warning(
                    "Vision refused or failed a credential request: %s (HTTP %s)",
                    error.code,
                    error.status,
                )
                if error.code == "unauthorized":
                    client = getattr(proxy.credentials, "client", None)
                    self._needs_action(
                        rejected_key_message(getattr(client, "key_source", None))
                    )
                elif error.code in {"not_permitted", "forbidden"}:
                    self._needs_action(f"cswap: {error}")
                elif error.code in {"rate_limited", "slow_down"}:
                    self._error(
                        429,
                        "cswap: Vision is limiting this key's requests; Claude "
                        "retries on its own.",
                        error.retry_after_seconds,
                    )
                else:
                    # An outage: native's own retries (about three minutes)
                    # ride out a short one without the person seeing it.
                    self._error(
                        503,
                        f"cswap: Vision could not be reached or failed "
                        f"({error.code}); Claude retries on its own. If this "
                        "persists, check https://vision.infinity.inc.",
                        error.retry_after_seconds,
                    )

            def _setup_failure(self, error):
                """A local cswap problem (roster, settings, identity): say which."""
                _log.warning("Credential request failed: %s", error)
                self._needs_action(f"cswap: {error}")

            def _rejected_login(self, credential):
                """The provider refused a login and Vision has no newer one."""
                command = proxy.credentials.relogin_command(credential)
                email = credential.get("email", "this account")
                _log.warning("The provider rejected the login for %s.", email)
                self._needs_action(
                    f"cswap: Anthropic rejected the login for {email}, and "
                    f"Vision has no newer one. Log it in again with `{command}`, "
                    "then retry."
                )

            def do_POST(self):
                self.connection.settimeout(30)
                self.close_connection = True
                expected = "Bearer " + proxy.capability
                provided = self.headers.get("Authorization", "")
                if not secrets.compare_digest(
                    provided.encode(), expected.encode()
                ) or self.headers.get("Origin"):
                    self._needs_action(STRAY_LOGIN_MESSAGE)
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
                except NoUsableLogin as error:
                    self._no_usable_login(error)
                    return
                except VisionError as error:
                    self._vision_failure(error)
                    return
                except ClaudeSwitchError as error:
                    self._setup_failure(error)
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
                            self._rejected_login(credential)
                            return
                        credential = replacement
                        response = proxy.open_message(
                            self.path, body, headers, credential
                        )
                        if response.status == 401:
                            # Replayed once already; a 401 passed through would
                            # make native say "Please run /login".
                            response.close()
                            self._rejected_login(credential)
                            return
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
                    self._vision_failure(error)
                except NoUsableLogin as error:
                    self._no_usable_login(error)
                except ClaudeSwitchError as error:
                    self._setup_failure(error)
                except (OSError, http.client.HTTPException) as error:
                    # URLError and timeouts, but also what http.client raises
                    # unwrapped while reading the reply: TLS alerts such as
                    # BAD_RECORD_MAC, resets, a hang-up, a truncated stream.
                    # After the reply started it may instead be native hanging
                    # up. No retry here: only the native client decides whether
                    # to replay a request whose upstream execution is uncertain.
                    _log.warning(
                        "Native request failed %s: %r",
                        "mid-reply" if response_started else "before a reply",
                        error,
                    )
                    if response_started:
                        self._reset()
                    else:
                        self._error(502, "The provider connection is unavailable.")

        self.server = _AdapterServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def _announce_relogin(self):
        """Print which logins need a re-login, before native starts drawing.

        The one moment the adapter may write to the terminal: once native
        runs, stderr is its screen. Bounded and never raises (P5).
        """
        from claude_swap.vision_pool import CentralPoolCredential

        if isinstance(self.credentials, CentralPoolCredential):
            for line in self.credentials.relogin_notices():
                print(line, file=sys.stderr)

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
