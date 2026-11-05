"""Mock OpenAI-compatible provider that streams the last user message back as deltas.

Test knobs (extra body fields, ignored by real providers):
  mock_chunk_max   max characters per delta (random 1..N, seeded)      default 6
  mock_seed        RNG seed for chunk boundaries                       default 0
  mock_delay_ms    delay between deltas                                default 0
  mock_ttft_ms     delay before the first delta                        default 0
  mock_mode        "content" | "tool_call" | "error_midstream" | "no_done" | "logprobs"
  n                number of choices (each echoes the text)

Run: uv run uvicorn --factory task3_llm_guardrail.mock_llm:app_factory --port 8301
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def _chunks(text: str, max_size: int, seed: int) -> list[str]:
    rnd, out, i = random.Random(seed), [], 0
    while i < len(text):
        n = rnd.randint(1, max(1, max_size))
        out.append(text[i : i + n])
        i += n
    return out


def build() -> tuple[FastAPI, dict[str, Any]]:
    app = FastAPI(title="Mock LLM provider")
    state: dict[str, Any] = {"requests": 0, "cancelled": 0, "completed": 0}

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        state["requests"] += 1
        text = next((m["content"] for m in reversed(body["messages"]) if m.get("role") == "user"), "")
        mode = body.get("mock_mode", "content")
        n = int(body.get("n", 1))
        base = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": int(time.time()), "model": body.get("model", "mock-1")}

        if not body.get("stream"):
            msg: dict[str, Any] = {"role": "assistant", "content": text}
            if mode == "tool_call":
                msg = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "send_email", "arguments": json.dumps({"body": text})}}
                ]}
            return JSONResponse({**base, "object": "chat.completion", "choices": [
                {"index": i, "message": msg, "finish_reason": "stop"} for i in range(n)
            ], "usage": {"prompt_tokens": 5, "completion_tokens": len(text) // 4, "total_tokens": 5 + len(text) // 4}})

        pieces = _chunks(text if mode != "tool_call" else json.dumps({"body": text}), body.get("mock_chunk_max", 6), body.get("mock_seed", 0))
        delay = body.get("mock_delay_ms", 0) / 1000
        ttft = body.get("mock_ttft_ms", 0) / 1000

        def frame(obj: dict) -> bytes:
            return f"data: {json.dumps(obj)}\n\n".encode()

        async def gen():
            try:
                if ttft:
                    await asyncio.sleep(ttft)
                for i in range(n):
                    first = {"role": "assistant", "content": ""}
                    if mode == "tool_call":
                        first = {"role": "assistant", "content": None, "tool_calls": [
                            {"index": 0, "id": "call_1", "type": "function", "function": {"name": "send_email", "arguments": ""}}
                        ]}
                    yield frame({**base, "choices": [{"index": i, "delta": first, "finish_reason": None}]})
                for k, piece in enumerate(pieces):
                    if mode == "error_midstream" and k == len(pieces) // 2:
                        yield frame({"error": {"message": "internal: shard 7 OOM at /srv/model.py:42", "type": "server_error"}})
                        return
                    for i in range(n):
                        if mode == "tool_call":
                            delta = {"tool_calls": [{"index": 0, "function": {"arguments": piece}}]}
                        else:
                            delta = {"content": piece}
                        choice: dict[str, Any] = {"index": i, "delta": delta, "finish_reason": None}
                        if mode == "logprobs":
                            choice["logprobs"] = {"content": [{"token": piece, "logprob": -0.1, "bytes": list(piece.encode())}]}
                        yield frame({**base, "choices": [choice]})
                    if delay:
                        await asyncio.sleep(delay)
                for i in range(n):
                    yield frame({**base, "choices": [{"index": i, "delta": {}, "finish_reason": "tool_calls" if mode == "tool_call" else "stop"}]})
                if mode == "no_done":
                    return
                yield b"data: [DONE]\n\n"
                state["completed"] += 1
            except asyncio.CancelledError:
                state["cancelled"] += 1
                raise

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app, state


def app_factory() -> FastAPI:
    return build()[0]
