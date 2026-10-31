"""Token usage read from a provider reply while the adapter forwards it.

The adapter writes each chunk to native Claude first and feeds it here after,
so reading usage never delays a byte. Nothing is buffered beyond what the
usage needs: a streamed reply keeps only the current line, and only while it
may belong to a ``message_start`` or ``message_delta`` event; a JSON reply is
kept up to a bound and then given up on.

Anthropic's usage: ``message_start`` carries the model and the input-side
counts, the last ``message_delta`` the final counts (its values win). A stream
counts only when it ended normally and reached ``message_delta``; an aborted
one reports no usage. ``input_tokens`` is already uncached input only, which is
what the ledger and Vision's usage report expect.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)
# The SSE events whose data lines carry usage; every other data line is skipped.
_USAGE_EVENTS = frozenset({b"message_start", b"message_delta"})
_MAX_KEPT_LINE = 1024 * 1024
_MAX_JSON_BODY = 8 * 1024 * 1024
_MAX_COUNT = 10**12
_MAX_MODEL_LENGTH = 256
_UNPARSED = object()


@dataclass(frozen=True)
class TokenUsage:
    """The provider's counts for one request; None where it gave none."""

    input_tokens: int | None
    output_tokens: int | None
    cache_read_input_tokens: int | None
    cache_creation_input_tokens: int | None


def _count(value: object) -> int | None:
    if type(value) is int and 0 <= value <= _MAX_COUNT:
        return value
    return None


def _model(value: object) -> str | None:
    if isinstance(value, str) and 0 < len(value) <= _MAX_MODEL_LENGTH:
        return value
    return None


def _counts(usage: object) -> dict[str, int | None]:
    if not isinstance(usage, dict):
        return {}
    return {field: _count(usage.get(field)) for field in TOKEN_FIELDS if field in usage}


class SseUsageReader:
    """Usage from an Anthropic ``text/event-stream`` reply, fed chunk by chunk."""

    def __init__(self) -> None:
        self._line = bytearray()
        self._skipping = False  # the rest of the current line is not needed
        self._event = b""
        self._model: str | None = None
        self._counts: dict[str, int | None] = {}
        self._final = False  # a message_delta with usage arrived

    def feed(self, chunk: bytes) -> None:
        start = 0
        size = len(chunk)
        while start < size:
            newline = chunk.find(b"\n", start)
            end = size if newline < 0 else newline
            if not self._skipping:
                self._line += chunk[start:end]
                if not self._wanted():
                    self._skipping = True
                    self._line.clear()
            if newline < 0:
                return
            if not self._skipping:
                self._handle(bytes(self._line).rstrip(b"\r"))
            self._line.clear()
            self._skipping = False
            start = newline + 1

    def _wanted(self) -> bool:
        """Whether the line being collected could carry usage."""
        line = self._line
        if len(line) > _MAX_KEPT_LINE:
            return False
        if line.startswith(b"data:"):
            return self._event in _USAGE_EVENTS
        # Undecided (shorter than the prefix) or an event/comment/blank line.
        return True

    def _handle(self, line: bytes) -> None:
        if not line:
            self._event = b""  # the blank line that ends an event
        elif line.startswith(b"event:"):
            self._event = line[6:].strip()
        elif line.startswith(b"data:") and self._event in _USAGE_EVENTS:
            try:
                data = json.loads(line[5:])
            except ValueError:
                return
            if not isinstance(data, dict):
                return
            if data.get("type") == "message_start":
                message = data.get("message")
                if isinstance(message, dict):
                    self._model = _model(message.get("model"))
                    self._counts.update(_counts(message.get("usage")))
            elif data.get("type") == "message_delta":
                delta = _counts(data.get("usage"))
                if "usage" in data:
                    self._final = True
                # A delta's explicit value wins; its nulls do not erase.
                self._counts.update(
                    {field: value for field, value in delta.items() if value is not None}
                )

    def model(self) -> str | None:
        return self._model

    def usage(self, *, complete: bool) -> TokenUsage | None:
        if not complete or not self._final:
            return None
        return TokenUsage(*(self._counts.get(field) for field in TOKEN_FIELDS))

    def buffered_bytes(self) -> int:
        return len(self._line)


class JsonUsageReader:
    """Usage from a JSON reply: a message, or a ``count_tokens`` answer."""

    def __init__(self, *, count_tokens: bool, limit: int = _MAX_JSON_BODY) -> None:
        self.count_tokens = count_tokens
        self._limit = limit
        self._body: bytearray | None = bytearray()
        self._value: object = _UNPARSED

    def feed(self, chunk: bytes) -> None:
        if self._body is None:
            return
        self._value = _UNPARSED
        if len(self._body) + len(chunk) > self._limit:
            self._body = None  # too large to keep; its usage goes unreported
            return
        self._body += chunk

    def _parsed(self) -> dict[str, object] | None:
        if self._body is None:
            return None
        if self._value is _UNPARSED:
            try:
                value = json.loads(self._body)
            except ValueError:
                value = None
            self._value = value if isinstance(value, dict) else None
        return self._value

    def model(self) -> str | None:
        parsed = self._parsed()
        return None if parsed is None else _model(parsed.get("model"))

    def usage(self, *, complete: bool) -> TokenUsage | None:
        parsed = self._parsed() if complete else None
        if parsed is None:
            return None
        if self.count_tokens:
            # count_tokens answers how many input tokens a request would use;
            # it generates nothing.
            counted = _count(parsed.get("input_tokens"))
            return None if counted is None else TokenUsage(counted, None, None, None)
        counts = _counts(parsed.get("usage"))
        if not counts:
            return None
        return TokenUsage(*(counts.get(field) for field in TOKEN_FIELDS))

    def buffered_bytes(self) -> int:
        return 0 if self._body is None else len(self._body)


UsageReader = SseUsageReader | JsonUsageReader


def usage_reader(
    path: str, status: int, headers: Mapping[str, str]
) -> UsageReader | None:
    """A reader for a successful, uncompressed reply that can carry usage."""
    if not 200 <= status < 300:
        return None
    encoding = (headers.get("Content-Encoding") or "identity").strip().lower()
    if encoding != "identity":
        return None
    content_type = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
    route = path.split("?", 1)[0]
    if content_type == "text/event-stream":
        return SseUsageReader()
    if content_type == "application/json":
        return JsonUsageReader(count_tokens=route == "/v1/messages/count_tokens")
    return None
