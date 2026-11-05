"""Task 3: streaming PII guardrail."""

from __future__ import annotations

import json
import random
import time

import httpx
import pytest

from task3_llm_guardrail.gateway import GuardSettings, create_app
from task3_llm_guardrail.mock_llm import build as build_mock
from task3_llm_guardrail.redactor import MAX_MATCH_LEN, REDACTED, StreamRedactor, luhn_valid, redact_text
from task3_llm_guardrail.sse import SSEDecoder

from .conftest import run_uvicorn


# =============================================================================== pattern unit tests
@pytest.mark.parametrize(
    "pii",
    [
        "jane.doe@example.com",
        "j+tag@sub.mail.example.co.uk",
        "a_b%c@x-y.io",
        "123-45-6789",
        "123 45 6789",
        "4111111111111111",
        "4111 1111 1111 1111",
        "4111-1111-1111-1111",
        "378282246310005",
        "3782 822463 10005",
        "5555555555554444",
        "6011111111111117",
    ],
)
def test_detects(pii):
    for template in ("{}", "x {} y", "({})", "{}.", '"{}"', "→{}←"):
        out, n = redact_text(template.format(pii))
        assert pii not in out and n == 1, (template, out)


@pytest.mark.parametrize(
    "benign",
    [
        "4111 1111 1111 1112",  # fails Luhn
        "order 1234567890123",  # 13 digits, not Luhn
        "000-12-3456",
        "666-12-3456",
        "123-00-4567",
        "123-45-0000",  # invalid SSN ranges (Presidio rules)
        "123-45 6789",  # mixed separators
        "call 555-123-4567",  # phone number
        "version 1.2.3",
        "2026-11-05",
        "user@localhost",
        "@handle",
        "not an email@",
        "1234-5678-9012-3456-7890",  # too long for any card shape
    ],
)
def test_does_not_over_redact(benign):
    out, n = redact_text(benign)
    assert out == benign and n == 0


def test_luhn():
    assert luhn_valid("4111111111111111") and not luhn_valid("4111111111111112")


# =============================================================================== streaming == batch
PII_SAMPLES = [
    "jane.doe@example.com",
    "123-45-6789",
    "4111 1111 1111 1111",
    "378282246310005",
    "a@b.co",
    "x.y@z.com.au",
]
FILLER = [
    "the",
    "order",
    "1234",
    "2024",
    "id",
    "café",
    "naïve",
    "😀",
    "-",
    "@",
    ".",
    ",",
    "\n",
    "a@",
    "4111",
    "123-45",
    "SSN",
    "ssn:",
    "social security number",
    "is",
    "123456789",
    "4111.1111",
    "–",
    "exämple",
]


def _random_text(rnd: random.Random, words: int = 60) -> str:
    parts = []
    for _ in range(words):
        parts.append(rnd.choice(PII_SAMPLES) if rnd.random() < 0.15 else rnd.choice(FILLER))
        parts.append(rnd.choice([" ", " ", " ", "", ", ", ". ", "\n"]))
    return "".join(parts)


def _stream(text: str, sizes) -> tuple[str, StreamRedactor]:
    r, out, i = StreamRedactor(), [], 0
    for n in sizes:
        if i >= len(text):
            break
        out.append(r.feed(text[i : i + n]))
        i += n
    if i < len(text):
        out.append(r.feed(text[i:]))
    out.append(r.flush())
    return "".join(out), r


@pytest.mark.parametrize("seed", range(300))
def test_chunking_invariance_property(seed):
    """For any chunking, streaming output == batch redaction of the whole text."""
    rnd = random.Random(seed)
    text = _random_text(rnd)
    expected, _ = redact_text(text)
    sizes = [rnd.choice([1, 1, 2, 3, 4, 7, 16, 50]) for _ in range(len(text))]
    got, _ = _stream(text, sizes)
    assert got == expected


def test_char_by_char_and_whole():
    text = "Mail jane.doe@example.com, SSN 123-45-6789, card 4111-1111-1111-1111, ok 4111-1111-1111-1112."
    expected, n = redact_text(text)
    assert n == 3
    assert _stream(text, [1] * len(text))[0] == expected
    assert _stream(text, [len(text)])[0] == expected


def test_holdback_is_minimal_and_streams_promptly():
    r = StreamRedactor()
    # A completed word followed by whitespace is released immediately
    assert r.feed("Hello there ") == "Hello there "
    # A token that *could* be the start of an email is held...
    assert r.feed("jane") == ""
    assert r.feed(".doe@exa") == ""
    # ...and once it can no longer grow it is released, redacted
    assert r.feed("mple.com is my mail") == f"{REDACTED} is my "
    assert r.pending == "mail"
    assert r.flush() == "mail"


def test_bounded_memory_on_long_stream():
    rnd = random.Random(7)
    r = StreamRedactor()
    total_in = 0
    worst_case = "x" * 5000 + "@" + "a" * 5000  # pathological unbroken token
    for _ in range(2000):
        piece = _random_text(rnd, 10)
        total_in += len(piece)
        r.feed(piece)
    for i in range(0, len(worst_case), 13):
        r.feed(worst_case[i : i + 13])
    r.flush()
    assert total_in > 100_000
    assert r.max_buffered <= MAX_MATCH_LEN + 16


def test_redaction_marker_not_reprocessed():
    out, _ = _stream("[REDACTED] stays [REDACTED]", [3] * 20)
    assert out == "[REDACTED] stays [REDACTED]"


# =============================================================================== SSE decoder
def _decode(chunks: list[bytes]) -> list[str]:
    d = SSEDecoder()
    events = [e.data for c in chunks for e in d.feed(c)]
    return events + [e.data for e in d.close()]


def test_sse_crlf_split_and_multibyte_split():
    raw = 'data: {"a":"é😀"}\r\n\r\n: keep-alive\r\n\r\ndata: line1\ndata: line2\n\n'.encode()
    expected = ['{"a":"é😀"}', "line1\nline2"]
    assert _decode([raw]) == expected
    # every possible split point, including inside \r\n and inside multi-byte UTF-8 sequences
    for i in range(1, len(raw)):
        assert _decode([raw[:i], raw[i:]]) == expected, i
    assert _decode([bytes([b]) for b in raw]) == expected


def test_sse_bare_cr_line_endings():
    assert _decode([b"data: x\r\rdata: y\r\r"]) == ["x", "y"]


# =============================================================================== end-to-end
@pytest.fixture(scope="module")
def stack():
    mock_app, mock_state = build_mock()
    with run_uvicorn(mock_app) as mock_url:
        with run_uvicorn(create_app(GuardSettings(upstream_base_url=f"{mock_url}/v1"))) as gw:
            yield {"gw": gw, "mock": mock_url, "state": mock_state}


def _events(resp_text: str) -> list:
    out = []
    for block in resp_text.split("\n\n"):
        data = "\n".join(line[6:] for line in block.split("\n") if line.startswith("data: "))
        if data:
            out.append(data if data == "[DONE]" else json.loads(data))
    return out


def _content(events: list, field: str = "content", choice: int = 0) -> str:
    text = ""
    for e in events:
        if isinstance(e, dict):
            for c in e.get("choices", []):
                if c["index"] == choice:
                    text += c["delta"].get(field) or ""
    return text


SECRET_TEXT = "Contact jane.doe@example.com or 4111 1111 1111 1111, SSN 123-45-6789. Thanks!"


def _post(stack, **extra) -> httpx.Response:
    body = {"model": "mock-1", "stream": True, "messages": [{"role": "user", "content": SECRET_TEXT}], **extra}
    return httpx.post(f"{stack['gw']}/v1/chat/completions", json=body, timeout=20)


@pytest.mark.parametrize("chunk_max", [1, 3, 8, 200])
def test_stream_is_redacted(stack, chunk_max):
    resp = _post(stack, mock_chunk_max=chunk_max, mock_seed=chunk_max)
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
    events = _events(resp.text)
    assert events[-1] == "[DONE]"
    assert _content(events) == redact_text(SECRET_TEXT)[0]
    for secret in ("jane.doe", "4111", "6789"):
        assert secret not in resp.text
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"  # role chunk passed through
    assert any(c.get("finish_reason") == "stop" for e in events if isinstance(e, dict) for c in e["choices"])


def test_tool_call_arguments_are_redacted(stack):
    resp = _post(stack, mock_mode="tool_call", mock_chunk_max=4)
    events = _events(resp.text)
    args = ""
    for e in events:
        if isinstance(e, dict):
            for c in e["choices"]:
                for call in c["delta"].get("tool_calls") or []:
                    args += call.get("function", {}).get("arguments") or ""
    assert json.loads(args) == {"body": redact_text(SECRET_TEXT)[0]}
    assert "jane.doe" not in resp.text


def test_multiple_choices_isolated(stack):
    events = _events(_post(stack, n=2, mock_chunk_max=3).text)
    expected = redact_text(SECRET_TEXT)[0]
    assert _content(events, choice=0) == expected
    assert _content(events, choice=1) == expected


def test_logprobs_stripped(stack):
    resp = _post(stack, mock_mode="logprobs", mock_chunk_max=3)
    assert "logprobs" not in resp.text and "jane" not in resp.text


def test_upstream_error_sanitised(stack):
    resp = _post(stack, mock_mode="error_midstream", mock_chunk_max=3)
    events = _events(resp.text)
    assert events[-1]["error"]["type"] == "upstream_error"
    assert "shard" not in resp.text and "/srv/model.py" not in resp.text


def test_truncated_upstream_reported(stack):
    events = _events(_post(stack, mock_mode="no_done").text)
    assert events[-1]["error"]["code"] == "upstream_truncated"
    assert _content(events) == redact_text(SECRET_TEXT)[0]  # held text still flushed, redacted


def test_non_streaming_redacted(stack):
    resp = _post(stack, stream=False)
    assert resp.json()["choices"][0]["message"]["content"] == redact_text(SECRET_TEXT)[0]
    assert resp.headers["x-guardrail-redactions"] == "3"


def test_upstream_unavailable_is_sanitised():
    app = create_app(GuardSettings(upstream_base_url="http://127.0.0.1:1/v1", connect_timeout=1))
    with run_uvicorn(app) as gw:
        resp = httpx.post(f"{gw}/v1/chat/completions", json={"messages": [], "stream": True})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "upstream_unavailable"
    assert "request_id" in resp.json()["error"]


def test_first_token_latency_not_degraded(stack):
    """TTFT through the gateway ≈ provider TTFT: first redacted delta arrives long before the stream ends."""
    text = "The quick brown fox jumps over the lazy dog. " * 6
    body = {"stream": True, "messages": [{"role": "user", "content": text}], "mock_chunk_max": 4, "mock_delay_ms": 15}
    start = time.perf_counter()
    first_content = None
    with httpx.stream("POST", f"{stack['gw']}/v1/chat/completions", json=body, timeout=20) as resp:
        for line in resp.iter_lines():
            if (
                first_content is None
                and line.startswith("data: {")
                and '"content":"' in line
                and '"content":""' not in line
            ):
                first_content = time.perf_counter() - start
    total = time.perf_counter() - start
    assert first_content is not None
    assert total > 0.8  # ~70 deltas x 15 ms
    assert first_content < 0.25, f"TTFT {first_content:.3f}s vs total {total:.3f}s"


def test_client_disconnect_cancels_upstream(stack):
    before = stack["state"]["cancelled"]
    body = {"stream": True, "messages": [{"role": "user", "content": "word " * 400}], "mock_delay_ms": 20}
    with httpx.stream("POST", f"{stack['gw']}/v1/chat/completions", json=body, timeout=20) as resp:
        for _ in resp.iter_bytes():
            break  # hang up after the first bytes
    deadline = time.monotonic() + 5
    while stack["state"]["cancelled"] == before and time.monotonic() < deadline:
        time.sleep(0.05)
    assert stack["state"]["cancelled"] == before + 1, "upstream generation must stop when the client leaves"


# =============================================================================== iteration 3 (adversarial review regressions)
from task3_llm_guardrail.stream_guard import ChunkGuard, redact_completion  # noqa: E402


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SSN-123-45-6789", "SSN-[REDACTED]"),
        ("123-45-6789-x", "[REDACTED]-x"),
        ("card:-4111111111111111", "card:-[REDACTED]"),
        ("4111.1111.1111.1111", "[REDACTED]"),
        ("4111\u20131111\u20131111\u20131111", "[REDACTED]"),
        ("4111 1111-1111 1111", "[REDACTED]"),
        ("john@ex\u00e4mple.com", "[REDACTED]"),
        ("\uff14" + "\uff11" * 15, "[REDACTED]"),  # full-width 4111111111111111, Luhn via unicodedata
        ("My SSN is 123456789.", "My SSN is [REDACTED]."),
        ("social security number: 123456789", "social security number: [REDACTED]"),
        ("order 123456789", "order 123456789"),  # unlabelled 9 digits: not an SSN
    ],
)
def test_detection_gaps_closed(text, expected):
    assert redact_text(text)[0] == expected


def _run_guard(chunks: list[dict]) -> list[dict]:
    g, out = ChunkGuard(), []
    for c in chunks:
        if (o := g.process(c)) is not None:
            out.append(o)
    if (tail := g.finish()) is not None:
        out.append(tail)
    return out


def _collect(out: list[dict], *path) -> str:
    text = ""
    for c in out:
        for ch in c["choices"]:
            node = ch
            try:
                for p in path:
                    node = node[p]
            except (KeyError, IndexError, TypeError):
                continue
            text += node if isinstance(node, str) else ""
    return text


def _split(s: str, n: int = 3) -> list[str]:
    return [s[i : i + n] for i in range(0, len(s), n)]


def test_legacy_function_call_arguments_redacted_and_intact():
    args = json.dumps({"to": "john.doe@example.com", "ssn": "123-45-6789"})
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "function_call": {"name": "send", "arguments": ""}}}]}
    ]
    chunks += [{"choices": [{"index": 0, "delta": {"function_call": {"arguments": p}}}]} for p in _split(args)]
    chunks += [{"choices": [{"index": 0, "delta": {}, "finish_reason": "function_call"}]}]
    got = _collect(_run_guard(chunks), "delta", "function_call", "arguments")
    assert json.loads(got) == {"to": "[REDACTED]", "ssn": "[REDACTED]"}


@pytest.mark.parametrize(
    "delta_key, make",
    [
        ("content", lambda p: [{"type": "text", "text": p}]),
        ("audio", lambda p: {"transcript": p}),
        ("some_future_field", lambda p: p),
    ],
)
def test_unknown_and_nested_text_fields_redacted(delta_key, make):
    text = "reach me at jane.doe@example.com or 4111 1111 1111 1111 thanks"
    chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", delta_key: make(p)}}]} for p in _split(text, 4)]
    chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    blob = json.dumps(_run_guard(chunks))
    assert "jane.doe" not in blob and "4111" not in blob and "[REDACTED]" in blob


def test_parallel_tool_calls_without_index_do_not_mix():
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"id": "a", "function": {"name": "f", "arguments": "x jo"}},
                            {"id": "b", "function": {"name": "g", "arguments": "hn@ex.com y"}},
                        ]
                    },
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]
    out = _run_guard(chunks)
    per_call: dict[str, str] = {}
    for c in out:
        for call in c["choices"][0]["delta"].get("tool_calls", []):
            per_call[call["id"]] = per_call.get(call["id"], "") + call.get("function", {}).get("arguments", "")
    # each call redacted independently: A keeps its own text, B's own email is redacted, nothing migrates
    assert per_call == {"a": "x jo", "b": "[REDACTED] y"}


def test_non_stream_redacts_every_field():
    body = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "mail a@b.com"}],
                    "function_call": {"name": "send", "arguments": '{"to": "x@y.org"}'},
                    "audio": {"transcript": "ssn 123-45-6789"},
                },
                "logprobs": {"content": [{"token": "a@b.com"}]},
            }
        ]
    }
    n = redact_completion(body)
    blob = json.dumps(body)
    assert n == 3 and "a@b.com" not in blob and "x@y.org" not in blob and "6789" not in blob and "logprobs" not in blob


def test_upstream_auth_failure_is_gateway_error():
    def handler(request):
        return httpx.Response(401, json={"error": {"message": "Incorrect API key provided: sk-abc***"}})

    app = create_app(GuardSettings(upstream_base_url="http://up/v1"), transport=httpx.MockTransport(handler))
    with run_uvicorn(app) as gw:
        r = httpx.post(f"{gw}/v1/chat/completions", json={"messages": [], "stream": True})
    assert r.status_code == 502 and "sk-abc" not in r.text


# =============================================================================== iteration 4 (peer benchmark)
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("john@example.com5", "[REDACTED]5"),  # TLDs are letters-only: a trailing digit ends the email
        ("john@example.com123-45-6789", "[REDACTED][REDACTED]"),
        ("94111 1111 1111 1111", "9[REDACTED]"),  # grouped card glued to a leading digit
        ("4111 1111 1111 11119", "[REDACTED]9"),  # ...and to a trailing digit (Luhn-valid prefix kept)
        ("123-45-67899", "[REDACTED]9"),
        ("a@b.com-foo", "[REDACTED]-foo"),
        ("a@b.example.com5.net ok", "[REDACTED] ok"),  # digits inside a label are fine
        ("4111 1111 1111 1112", "4111 1111 1111 1112"),  # still Luhn-gated
        ("call 555-123-4567 on 2026-11-05", "call 555-123-4567 on 2026-11-05"),
    ],
)
def test_digit_and_letter_adjacency(text, expected):
    assert redact_text(text)[0] == expected


# Ground-truth oracle (independent of the production regexes): we know exactly which PII strings were
# injected, so the streamed output must contain none of them, whatever glue surrounds them and however
# the stream is chunked. Contiguous card numbers glued to *digits* are excluded by design: an unseparated
# run inside a longer number is treated as an id (documented in the README's known-misses list).
ORACLE_PII = {
    "email": ["jane.doe@example.com", "j+tag@sub.mail.example.co.uk", "x_y@ex-ample.io"],
    "ssn": ["123-45-6789", "321 54 9876", "555.12.3456"],
    "card_grouped": ["4111 1111 1111 1111", "5555-5555-5555-4444", "3782 822463 10005", "6011.1111.1111.1117"],
    "card_plain": ["4111111111111111", "378282246310005"],
}
GLUE = ["", " ", ",", ".", "-", ":", "(", ")", "\n", "x", "Z", "9", "_", "é", "😀", '"']


@pytest.mark.parametrize("seed", range(250))
def test_ground_truth_oracle_no_injected_pii_survives(seed):
    rnd = random.Random(10_000 + seed)
    parts, injected = [], []
    for _ in range(rnd.randint(3, 8)):
        kind = rnd.choice(list(ORACLE_PII))
        pii = rnd.choice(ORACLE_PII[kind])
        before, after = rnd.choice(GLUE), rnd.choice(GLUE)
        if kind == "card_plain":
            before = before if not before.isdigit() else " "
            after = after if not after.isdigit() else " "
        if kind == "email" and (before in "._-" or before.isalnum() or before == "é"):
            before = " "  # would legitimately extend the local part; the email is still redacted, just longer
        parts.append(f"{rnd.choice(FILLER)} {before}{pii}{after} {rnd.choice(FILLER)}")
        injected.append(pii)
    text = " ".join(parts)
    sizes = [rnd.choice([1, 2, 3, 5, 8, 13]) for _ in range(len(text))]
    got, _ = _stream(text, sizes)
    for pii in injected:
        assert pii not in got, (pii, text, got)


def test_heartbeat_keeps_idle_stream_alive():
    """Upstream silent for 1.2 s before its first token -> client sees ': ping' comments, then content."""
    mock_app, _ = build_mock()
    with run_uvicorn(mock_app) as m, run_uvicorn(create_app(GuardSettings(f"{m}/v1", heartbeat_s=0.3))) as gw:
        body = {"stream": True, "messages": [{"role": "user", "content": "hi jane@x.org"}], "mock_ttft_ms": 1200}
        text = httpx.post(f"{gw}/v1/chat/completions", json=body, timeout=10).text
    assert text.count(": ping") >= 2
    assert text.index(": ping") < text.index('"content"')
    assert "jane@x.org" not in text and _events(text)[-1] == "[DONE]"
