"""Measure what reading token usage costs a streamed reply.

Two measurements:

1. Parser cost: ``SseUsageReader.feed`` time per chunk, on synthetic Anthropic
   streams delivered one event per chunk (how the provider flushes) and in
   64 KiB chunks (the adapter's read size).
2. End to end: the same stream through the real adapter from a loopback
   provider that flushes one event at a time, with usage reading on and off.
   Reported per chunk: the time from the provider writing a chunk to the
   client receiving it.

The adapter feeds each chunk to the parser only after writing it to native,
so (2) is what native sees; (1) bounds the work moved after each write.

Run: uv run python scripts/bench_usage_reader.py
"""

from __future__ import annotations

import http.server
from contextlib import nullcontext
import json
import statistics
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from unittest.mock import Mock, patch

from claude_swap.account_ledger import AccountLedger
from claude_swap.response_usage import SseUsageReader
from claude_swap.usage_totals import UsageTotals
from claude_swap.vision_proxy import InferenceProxy


def event(name: str, payload: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()


def stream(text_events: int, *, tool_json_bytes: int = 0) -> list[bytes]:
    events = [
        event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "model": "claude-opus-4-6",
                    "usage": {
                        "input_tokens": 12,
                        "cache_creation_input_tokens": 300,
                        "cache_read_input_tokens": 40000,
                        "output_tokens": 1,
                    },
                },
            },
        )
    ]
    for index in range(text_events):
        events.append(
            event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": f"token {index} of the reply "},
                },
            )
        )
    if tool_json_bytes:
        events.append(
            event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": "x" * tool_json_bytes},
                },
            )
        )
    events.append(
        event(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 900}},
        )
    )
    events.append(event("message_stop", {"type": "message_stop"}))
    return events


def rechunk(events: list[bytes], size: int) -> list[bytes]:
    data = b"".join(events)
    return [data[offset : offset + size] for offset in range(0, len(data), size)]


def parser_cost(chunks: list[bytes], repeats: int = 200) -> float:
    """Median nanoseconds of feed() per chunk."""
    samples = []
    for _ in range(repeats):
        reader = SseUsageReader()
        started = time.perf_counter_ns()
        for chunk in chunks:
            reader.feed(chunk)
        samples.append((time.perf_counter_ns() - started) / len(chunks))
        assert reader.usage(complete=True) is not None
    return statistics.median(samples)


def end_to_end(chunks: list[bytes], *, reading: bool, runs: int = 30) -> list[float]:
    """Microseconds from the provider's write of each chunk to the client's read."""
    written: list[float] = []

    class Provider(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for chunk in chunks:
                written.append(time.perf_counter())
                self.wfile.write(chunk)
                self.wfile.flush()
                time.sleep(0.0005)  # the provider paces events

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    credentials = Mock()
    credentials.get.return_value = {
        "accessToken": "synthetic",
        "login_id": "ail_bench",
        "account_id": "aia_bench",
        "email": "bench@infinity.inc",
    }
    credentials.account_alias.return_value = "bench"
    delays: list[float] = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        proxy = InferenceProxy(
            credentials,
            upstream=f"http://127.0.0.1:{server.server_port}",
            ledger=AccountLedger(root / "ledger"),
            usage_totals=UsageTotals(root / "totals.json", root / "machine_id"),
        )
        target = "claude_swap.vision_proxy.usage_reader"
        switch = nullcontext() if reading else patch(target, return_value=None)
        with proxy, switch:
            for _ in range(runs):
                written.clear()
                received: list[float] = []
                request = urllib.request.Request(
                    proxy.url + "/v1/messages",
                    data=b'{"model":"claude-opus-4-6","stream":true}',
                    headers={"Authorization": "Bearer " + proxy.capability},
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    expected = b"".join(chunks)
                    got = b""
                    for chunk in chunks:
                        part = b""
                        while len(part) < len(chunk):
                            part += response.read1(len(chunk) - len(part))
                        received.append(time.perf_counter())
                        got += part
                    assert got == expected
                delays.extend(
                    (arrival - sent) * 1e6 for sent, arrival in zip(written, received)
                )
    server.shutdown()
    server.server_close()
    return delays


def main() -> None:
    text = stream(400)
    tool = stream(50, tool_json_bytes=2_000_000)
    print("Parser cost, median ns per chunk (feed only):")
    print(f"  text stream, one event per chunk ({len(text)} chunks): {parser_cost(text):8.0f} ns")
    big = rechunk(text, 65536)
    print(f"  text stream, 64 KiB chunks ({len(big)} chunks):          {parser_cost(big):8.0f} ns")
    tool_chunks = rechunk(tool, 65536)
    print(
        f"  2 MB tool-input stream, 64 KiB chunks ({len(tool_chunks)} chunks): "
        f"{parser_cost(tool_chunks, repeats=50):8.0f} ns"
    )
    events = stream(200)
    print("End to end through the adapter, per chunk, provider write -> client read (us):")
    for reading in (False, True):
        delays = sorted(end_to_end(events, reading=reading))
        label = "usage reading on " if reading else "usage reading off"
        print(
            f"  {label}: median {statistics.median(delays):6.1f}  "
            f"p90 {delays[int(len(delays) * 0.9)]:6.1f}  "
            f"p99 {delays[int(len(delays) * 0.99)]:6.1f}  (n={len(delays)})"
        )


if __name__ == "__main__":
    main()
