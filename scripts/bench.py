"""Reproducible numbers for the README.   uv run python scripts/bench.py [--quick]

1. Redactor CPU cost per delta (p50 / p99), realistic and pathological text.
2. Time-to-first-token added by the Task 3 guardrail vs. calling the provider directly (p50 / p95),
   and the worst case: a long email *opening* the stream must be held until it is complete.
3. Task 4 deadline precision: N concurrent requests to a provider that never answers in time;
   when does failover actually fire relative to the 3000 ms budget?
Writes docs/BENCHMARKS.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from task3_llm_guardrail.redactor import StreamRedactor, redact_text  # noqa: E402
from tests.conftest import run_uvicorn  # noqa: E402

LINES: list[str] = []


def emit(line: str = "") -> None:
    print(line)
    LINES.append(line)


def pct(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


# ---------------------------------------------------------------------------------------------- 1
def bench_redactor(n_words: int) -> None:
    rnd = random.Random(1)
    words = ["the", "model", "said", "that", "order", "2024", "café", "results", "were", "good", "and", "value"]
    pii = ["jane.doe@example.com", "123-45-6789", "4111 1111 1111 1111"]
    cases = {
        "realistic text (2% PII)": " ".join(
            rnd.choice(pii) if rnd.random() < 0.02 else rnd.choice(words) for _ in range(n_words)
        ),
        "pathological: 200 KB digit run": "1" * 200_000,
        "pathological: 200 KB letter run": "a" * 200_000,
    }
    emit("### 1. Redactor cost per delta (4-character deltas ≈ 1 token)")
    emit()
    emit("| input | deltas | p50 µs | p99 µs | max held chars | stream == batch |")
    emit("|---|---:|---:|---:|---:|:-:|")
    for name, text in cases.items():
        chunks = [text[i : i + 4] for i in range(0, len(text), 4)]
        r, times, out = StreamRedactor(), [], []
        for c in chunks:
            t = time.perf_counter_ns()
            out.append(r.feed(c))
            times.append((time.perf_counter_ns() - t) / 1000)
        out.append(r.flush())
        same = "".join(out) == redact_text(text)[0]
        emit(
            f"| {name} | {len(chunks):,} | {pct(times, 50):.1f} | {pct(times, 99):.1f} | {r.max_buffered} | {'✓' if same else '✗'} |"
        )
    emit()


# ---------------------------------------------------------------------------------------------- 2
async def _ttft(client: httpx.AsyncClient, url: str, body: dict) -> float:
    t0 = time.perf_counter()
    async with client.stream("POST", url, json=body) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: {"):
                chunk = json.loads(line[6:])
                if any((c.get("delta") or {}).get("content") for c in chunk.get("choices", [])):
                    return (time.perf_counter() - t0) * 1000
    return float("nan")


async def bench_ttft(runs: int) -> None:
    from task3_llm_guardrail.gateway import GuardSettings, create_app
    from task3_llm_guardrail.mock_llm import build

    mock, _ = build()
    with run_uvicorn(mock) as m, run_uvicorn(create_app(GuardSettings(f"{m}/v1"))) as g:
        async with httpx.AsyncClient(timeout=30) as c:
            normal = {
                "stream": True,
                "messages": [{"role": "user", "content": "Hello there, here is your answer. " * 4}],
                "mock_chunk_max": 4,
                "mock_delay_ms": 10,
            }
            email_first = {
                **normal,
                "messages": [
                    {"role": "user", "content": "firstname.lastname.department@subsidiary.example.com is the contact."}
                ],
            }
            rows = []
            for label, url, body in [
                ("provider direct (no guardrail)", f"{m}/v1/chat/completions", normal),
                ("through guardrail", f"{g}/v1/chat/completions", normal),
                ("through guardrail, 50-char email opens the stream", f"{g}/v1/chat/completions", email_first),
            ]:
                for _ in range(3):  # warm-up
                    await _ttft(c, url, body)
                samples = [await _ttft(c, url, body) for _ in range(runs)]
                rows.append((label, pct(samples, 50), pct(samples, 95)))
    emit(f"### 2. Time to first content token ({runs} runs each; provider emits a 4-char delta every 10 ms)")
    emit()
    emit("| path | p50 ms | p95 ms |")
    emit("|---|---:|---:|")
    for label, p50, p95 in rows:
        emit(f"| {label} | {p50:.1f} | {p95:.1f} |")
    emit()
    emit(
        "The guardrail adds about one held-back word. The worst case is inherent to exact redaction: text that"
        " might still turn out to be an email can't be released before it's known not to be one. The"
        " bound is the PII length, not the response length."
    )
    emit()


# ---------------------------------------------------------------------------------------------- 3
async def bench_deadline(concurrency: int) -> None:
    from task4_model_router.app import RouterSettings, create_app
    from task4_model_router.mock_providers import build
    from task4_model_router.router import Provider

    p_app, p = build("primary")
    s_app, _ = build("secondary")
    p.update(mode="slow", delay_s=10.0)
    with tempfile.TemporaryDirectory() as tmp, run_uvicorn(p_app) as pu, run_uvicorn(s_app) as su:
        settings = RouterSettings(
            db_path=f"{tmp}/b.db",
            primary=Provider("primary", f"{pu}/v1", timeout_s=3.0),
            secondary=Provider("secondary", f"{su}/v1", timeout_s=10.0),
            tenants=[("k", "bench", 10_000_000)],
        )
        with run_uvicorn(create_app(settings)) as g:
            async with httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_connections=concurrency * 2)) as c:

                async def one() -> tuple[float, str]:
                    t0 = time.perf_counter()
                    r = await c.post(
                        f"{g}/v1/chat/completions",
                        headers={"authorization": "Bearer k"},
                        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
                    )
                    return (time.perf_counter() - t0) * 1000, r.headers.get("x-gateway-provider", "?")

                results = await asyncio.gather(*(one() for _ in range(concurrency)))
    times = [t for t, _ in results]
    served = {prov for _, prov in results}
    emit(f"### 3. Failover deadline precision ({concurrency} concurrent requests, primary takes 10 s)")
    emit()
    emit("| min ms | p50 ms | p95 ms | max ms | served by |")
    emit("|---:|---:|---:|---:|---|")
    emit(
        f"| {min(times):.0f} | {pct(times, 50):.0f} | {pct(times, 95):.0f} | {max(times):.0f} | {', '.join(sorted(served))} |"
    )
    emit()
    emit(
        "Budget: 3000 ms. The excess over 3000 ms is cancellation, then the backup round-trip, then"
        " SQLite reserve and reconcile. Nothing waits for the slow primary."
    )
    emit()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    emit("# Benchmarks")
    emit()
    emit(
        f"Generated by `uv run python scripts/bench.py` on Python {sys.version.split()[0]}. Local loopback, single machine."
    )
    emit()
    bench_redactor(50_000 if args.quick else 200_000)
    asyncio.run(bench_ttft(10 if args.quick else 40))
    asyncio.run(bench_deadline(20 if args.quick else 50))
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "BENCHMARKS.md").write_text("\n".join(LINES) + "\n")
    print("wrote docs/BENCHMARKS.md")


if __name__ == "__main__":
    main()
