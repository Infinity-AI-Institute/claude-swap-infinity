"""Who owns the Vision API key this process uses, for the account ledger.

Vision describes the calling key at ``GET /api/api-keys/self`` (capability
``self:read``, which every key holds and cannot lose): ``data.owner.email`` is
the Vision account of the person who minted it. That person may differ from
the shared Claude account serving a request, which is why the ledger records
both.

The lookup runs on a background thread: once when the adapter starts, again
whenever the client adopts a re-read key after Vision rejected the old one
(``VisionClient.key_generation`` moves), and again at most once a minute after
a failure. A request only reads the cached answer and is never delayed; until
an answer for the key in use arrives it records null.

The address is a person's: it is never logged.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from claude_swap.account_ledger import person_email
from claude_swap.exceptions import ClaudeSwitchError
from claude_swap.vision import VisionClient

SELF_PATH = "/api/api-keys/self"
RETRY_AFTER_FAILURE_SECONDS = 60

# Runs inside the native adapter: cswap's log file only, never stderr.
_log = logging.getLogger("claude-swap.vision_key_owner")
_log.addHandler(logging.NullHandler())


def owner_email(answer: object) -> str | None:
    """``owner.email`` from an api-keys/self answer about a key, else None."""
    if not isinstance(answer, dict) or answer.get("via") != "api_key":
        return None
    owner = answer.get("owner")
    if not isinstance(owner, dict):
        return None
    return person_email(owner.get("email"))


class KeyOwnerLookup:
    """The cached owner email of ``client``'s current key; None while unknown."""

    def __init__(
        self, client: VisionClient, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.client = client
        self.clock = clock
        self._lock = threading.Lock()
        self._email: str | None = None
        self._resolved_generation: int | None = None
        self._in_flight = False
        self._retry_at: float | None = None

    def start(self) -> None:
        """Begin the first lookup without waiting for it."""
        self.current()

    def resolving(self) -> bool:
        with self._lock:
            return self._in_flight

    def current(self) -> str | None:
        """The owner of the key in use if already known; never blocks on Vision."""
        generation = self.client.key_generation
        with self._lock:
            if self._resolved_generation == generation:
                return self._email
            if not self._in_flight and (
                self._retry_at is None or self.clock() >= self._retry_at
            ):
                self._in_flight = True
                threading.Thread(
                    target=self._resolve, name="cswap-key-owner", daemon=True
                ).start()
            return None

    def _resolve(self) -> None:
        generation = self.client.key_generation
        email: str | None = None
        failed = True
        try:
            email = owner_email(self.client.request("GET", SELF_PATH))
            failed = False
        except ClaudeSwitchError as error:
            # VisionError's text names only Vision's origin and the refusal
            # code, never the key or an address.
            _log.warning("Could not look up the Vision API key's owner: %s", error)
        except Exception:
            # A thread's uncaught exception is printed to stderr, which is
            # native Claude's screen; a bug here must only cost the field.
            _log.warning("Vision API key owner lookup failed unexpectedly.", exc_info=True)
        finally:
            with self._lock:
                self._in_flight = False
                if failed:
                    self._retry_at = self.clock() + RETRY_AFTER_FAILURE_SECONDS
                elif self.client.key_generation == generation:
                    self._email = email
                    self._resolved_generation = generation
                    self._retry_at = None
                # Otherwise the key was replaced mid-lookup; the next read asks again.
