"""Controllable mock OpenAI-compatible provider for exercising failover paths.

``state["mode"]`` (mutable at runtime by tests):
  ok                 normal JSON / SSE response
  rate_limited       429 with Retry-After (``state["retry_after"]``)
  slow               sleep ``state["delay_s"]`` before responding
  trickle            send headers immediately, then 1 byte every 0.5 s (defeats per-read timeouts)
  server_error       500 with a body full of internals (must never leak)
  bad_request        400
  stream_slow_first  SSE: role chunk at once, first content after ``delay_s``
  stream_break       SSE: one content chunk, then the connection drops
  stream_bad_bytes   SSE: invalid UTF-8 before any content
  stream_error_first SSE: role chunk, then an in-stream {"error": ...} before any content
  list_no_usage      JSON: list-valued content and no usage block

Run: uv run uvicorn --factory task4_model_router.mock_providers:app_factory --port 8401
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

LEAKY_BODY = {
    "error": {
        "message": "Traceback (most recent call last): File \"/opt/provider/internal/shard_router.py\", line 88 ... "
        "KeyError: org-SECRET-123 api_key=sk-live-SUPERSECRET",
        "type": "server_error",
    }
}


def build(name: str) -> tuple[FastAPI, dict[str, Any]]:
    app = FastAPI()
    state: dict[str, Any] = {"mode": "ok", "delay_s": 0.0, "retry_after": "2", "calls": 0, "name": name}

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> Response:
        body = await request.json()
        state["calls"] += 1
        state["last_body"] = body
        mode = state["mode"]
        text = f"Hello from {name}"
        prompt_tokens, completion_tokens = 20, 5

        if mode == "rate_limited":
            return JSONResponse(LEAKY_BODY, status_code=429, headers={"retry-after": state["retry_after"]})
        if mode == "server_error":
            return JSONResponse(LEAKY_BODY, status_code=500)
        if mode == "bad_request":
            return JSONResponse(LEAKY_BODY, status_code=400)
        if mode == "slow":
            await asyncio.sleep(state["delay_s"])
        if mode == "trickle":
            async def trickle():
                for _ in range(20):
                    yield b" "
                    await asyncio.sleep(0.5)
            return StreamingResponse(trickle(), media_type="application/json")

        base = {"id": f"chatcmpl-{name}", "created": int(time.time()), "model": body.get("model") or name}
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}

        if not body.get("stream"):
            if mode == "list_no_usage":
                return JSONResponse({**base, "object": "chat.completion", "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
                     "finish_reason": "stop"}]})
            return JSONResponse({**base, "object": "chat.completion", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            ], "usage": usage})

        def frame(obj: dict) -> bytes:
            return f"data: {json.dumps(obj)}\n\n".encode()

        async def gen():
            chunk = {**base, "object": "chat.completion.chunk"}
            if mode == "stream_bad_bytes":
                yield b"data: \xff\xfe\n\n"
                return
            yield frame({**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
            if mode == "stream_error_first":
                yield frame({"error": {"message": "internal: model shard 3 crashed", "type": "server_error"}})
                return
            if mode == "stream_slow_first":
                await asyncio.sleep(state["delay_s"])
            for word in text.split(" "):
                yield frame({**chunk, "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}]})
                if mode == "stream_break":
                    raise RuntimeError("connection lost")  # uvicorn aborts the response mid-stream
            yield frame({**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            if (body.get("stream_options") or {}).get("include_usage"):
                yield frame({**chunk, "choices": [], "usage": usage})
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app, state


def app_factory() -> FastAPI:
    import os

    return build(os.environ.get("MOCK_PROVIDER_NAME", "mock"))[0]
