"""Token usage read from provider responses as they pass through the adapter."""

import json

import pytest

from claude_swap.response_usage import (
    JsonUsageReader,
    SseUsageReader,
    TokenUsage,
    usage_reader,
)


def event(name, payload):
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()


START = event(
    "message_start",
    {
        "type": "message_start",
        "message": {
            "id": "msg_synthetic",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-4-6",
            "content": [],
            "usage": {
                "input_tokens": 12,
                "cache_creation_input_tokens": 300,
                "cache_read_input_tokens": 4000,
                "output_tokens": 1,
            },
        },
    },
)
DELTA = event(
    "content_block_delta",
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "hello"},
    },
)
FINAL = event(
    "message_delta",
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 57},
    },
)
STOP = event("message_stop", {"type": "message_stop"})
STREAM = START + event("ping", {"type": "ping"}) + DELTA * 3 + FINAL + STOP
EXPECTED = TokenUsage(
    input_tokens=12,
    output_tokens=57,
    cache_read_input_tokens=4000,
    cache_creation_input_tokens=300,
)


def feed_in(reader, data, size):
    for offset in range(0, len(data), size):
        reader.feed(data[offset : offset + size])
    return reader


@pytest.mark.parametrize("size", [1, 2, 3, 7, 64, 65536])
def test_a_stream_split_anywhere_yields_its_final_usage(size):
    reader = feed_in(SseUsageReader(), STREAM, size)
    assert reader.model() == "claude-opus-4-6"
    assert reader.usage(complete=True) == EXPECTED


def test_crlf_line_endings_are_accepted():
    reader = feed_in(SseUsageReader(), STREAM.replace(b"\n", b"\r\n"), 5)
    assert reader.usage(complete=True) == EXPECTED


def test_message_delta_values_override_message_start_values():
    final = event(
        "message_delta",
        {
            "type": "message_delta",
            "usage": {"output_tokens": 9, "input_tokens": 20, "cache_read_input_tokens": None},
        },
    )
    reader = feed_in(SseUsageReader(), START + final + STOP, 64)
    assert reader.usage(complete=True) == TokenUsage(20, 9, 4000, 300)


def test_an_aborted_stream_has_a_model_but_no_usage():
    reader = feed_in(SseUsageReader(), START + DELTA, 64)
    assert reader.model() == "claude-opus-4-6"
    assert reader.usage(complete=False) is None
    # Even a stream that reached message_delta is unreported if it did not end.
    reader = feed_in(SseUsageReader(), START + FINAL, 64)
    assert reader.usage(complete=False) is None


def test_a_stream_that_ends_without_message_delta_has_no_usage():
    error = event(
        "error",
        {"type": "error", "error": {"type": "overloaded_error", "message": "x"}},
    )
    reader = feed_in(SseUsageReader(), START + DELTA + error, 64)
    assert reader.usage(complete=True) is None


def test_a_huge_uninteresting_line_is_skipped_not_kept():
    huge = event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": "x" * 5_000_000},
        },
    )
    reader = feed_in(SseUsageReader(), START + huge + FINAL + STOP, 65536)
    assert reader.usage(complete=True) == EXPECTED
    assert reader.buffered_bytes() < 1024


def test_malformed_or_hostile_usage_values_become_null():
    start = event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "model": 42,
                "usage": {
                    "input_tokens": -1,
                    "cache_read_input_tokens": True,
                    "cache_creation_input_tokens": "300",
                    "output_tokens": 1,
                },
            },
        },
    )
    broken = b"event: message_delta\ndata: {not json\n\n"
    reader = feed_in(SseUsageReader(), start + broken + FINAL + STOP, 64)
    assert reader.model() is None
    assert reader.usage(complete=True) == TokenUsage(None, 57, None, None)


def test_a_non_streaming_message_reports_its_usage():
    body = json.dumps(
        {
            "id": "msg_synthetic",
            "model": "claude-sonnet-4-6",
            "usage": {
                "input_tokens": 5,
                "output_tokens": 6,
                "cache_read_input_tokens": 7,
                "cache_creation_input_tokens": 8,
            },
        }
    ).encode()
    reader = feed_in(JsonUsageReader(count_tokens=False), body, 10)
    assert reader.model() == "claude-sonnet-4-6"
    assert reader.usage(complete=True) == TokenUsage(5, 6, 7, 8)
    assert reader.usage(complete=False) is None


def test_count_tokens_reports_input_tokens_only():
    reader = feed_in(JsonUsageReader(count_tokens=True), b'{"input_tokens": 1234}', 4)
    assert reader.model() is None
    assert reader.usage(complete=True) == TokenUsage(1234, None, None, None)


def test_an_oversized_json_body_is_given_up_on():
    reader = JsonUsageReader(count_tokens=False, limit=100)
    reader.feed(b'{"usage": {"input_tokens": 1}, "pad": "' + b"x" * 200 + b'"}')
    assert reader.usage(complete=True) is None
    assert reader.buffered_bytes() == 0


@pytest.mark.parametrize(
    "path,status,content_type,encoding,expected",
    [
        ("/v1/messages?beta=true", 200, "text/event-stream; charset=utf-8", None, SseUsageReader),
        ("/v1/messages", 200, "application/json", None, JsonUsageReader),
        ("/v1/messages/count_tokens?beta=true", 200, "application/json", None, JsonUsageReader),
        ("/v1/messages", 400, "application/json", None, type(None)),
        ("/v1/messages", 529, "text/event-stream", None, type(None)),
        ("/v1/messages", 200, "text/event-stream", "gzip", type(None)),
        ("/v1/messages", 200, "text/html", None, type(None)),
    ],
)
def test_a_reader_is_chosen_only_for_a_successful_readable_reply(
    path, status, content_type, encoding, expected
):
    headers = {"Content-Type": content_type}
    if encoding:
        headers["Content-Encoding"] = encoding
    reader = usage_reader(path, status, headers)
    assert type(reader) is expected
    if isinstance(reader, JsonUsageReader):
        assert reader.count_tokens == ("count_tokens" in path)
