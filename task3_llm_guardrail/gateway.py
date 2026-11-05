"""Task 3 - LLM gateway with a real-time streaming PII guardrail.

    client ──POST /v1/chat/completions──► gateway ──► provider (OpenAI-compatible)
             ◄──── SSE, redacted ─────────  │  ◄──── SSE deltas
                                            └─ SSEDecoder → ChunkGuard (per-field StreamRedactor) → re-encode

Design points
* Zero full-response buffering: each upstream network read is decoded, redacted and
  written out before the next is awaited; per-field holdback is bounded (see redactor.py).
* Fail closed: if anything in the guard path raises, the client gets an in-stream
  ``guardrail_error`` event and the stream ends. Unredacted text is never "let through"
  (LiteLLM's Presidio hook falls back to yielding raw chunks on error; we don't).
* Upstream errors are sanitised: no provider bodies, org ids or stack traces reach the client.
* Client disconnect cancels the generator → ``finally`` closes the upstream response, so the
  provider stops generating (and billing).

Run:  GUARD_UPSTREAM_BASE_URL=https://api.openai.com/v1 GUARD_UPSTREAM_API_KEY=sk-... \
      uv run uvicorn --factory task3_llm_guardrail.gateway:app_factory --port 8300
      (defaults to the bundled mock provider at http://127.0.0.1:8301/v1)
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .sse import SSEDecoder, encode_event, with_heartbeat
from .stream_guard import ChunkGuard, redact_completion

logger = logging.getLogger("task3.guardrail")


@dataclass
class GuardSettings:
    upstream_base_url: str = "http://127.0.0.1:8301/v1"
    upstream_api_key: str | None = None
    connect_timeout: float = 5.0
    # Max gap between upstream chunks (httpx `read` is per-read, not total) - a stalled provider is cut off.
    read_timeout: float = 60.0
    max_body_bytes: int = 2_000_000
    # Idle gap after which an SSE ": ping" comment is sent (keeps proxies/LBs from cutting quiet streams).
    heartbeat_s: float | None = 15.0

    @classmethod
    def from_env(cls) -> GuardSettings:
        return cls(
            upstream_base_url=os.environ.get("GUARD_UPSTREAM_BASE_URL", cls.upstream_base_url),
            upstream_api_key=os.environ.get("GUARD_UPSTREAM_API_KEY"),
            heartbeat_s=float(os.environ.get("GUARD_HEARTBEAT_S", 15.0)) or None,
        )


def _error(status: int, message: str, err_type: str, code: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": err_type, "code": code, "param": None, "request_id": request_id}},
        status_code=status,
        headers={"x-request-id": request_id},
    )


def _error_event(message: str, err_type: str, code: str, request_id: str) -> bytes:
    payload = {"error": {"message": message, "type": err_type, "code": code, "param": None, "request_id": request_id}}
    return encode_event(json.dumps(payload))


def create_app(settings: GuardSettings, *, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    client = httpx.AsyncClient(
        transport=transport,
        base_url=settings.upstream_base_url.rstrip("/") + "/",
        timeout=httpx.Timeout(connect=settings.connect_timeout, read=settings.read_timeout, write=30.0, pool=5.0),
        limits=httpx.Limits(max_connections=500, max_keepalive_connections=100),
        follow_redirects=False,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await client.aclose()

    # NB: deliberately no GZipMiddleware - compression buffers SSE and wrecks TTFT.
    app = FastAPI(title="LLM Gateway - Streaming PII Guardrail", lifespan=lifespan)

    def upstream_headers() -> dict[str, str]:
        h = {"content-type": "application/json", "accept-encoding": "identity"}
        if settings.upstream_api_key:
            h["authorization"] = f"Bearer {settings.upstream_api_key}"  # gateway credential, never the client's
        return h

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        rid = f"req_{uuid.uuid4().hex}"
        t0 = time.perf_counter()
        raw = await request.body()
        if len(raw) > settings.max_body_bytes:
            return _error(413, "Request body too large", "invalid_request_error", "body_too_large", rid)
        try:
            body = json.loads(raw)
        except ValueError:
            return _error(400, "Request body must be valid JSON", "invalid_request_error", "invalid_json", rid)
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return _error(400, "'messages' must be an array", "invalid_request_error", "invalid_messages", rid)

        stream = bool(body.get("stream"))
        req = client.build_request("POST", "chat/completions", json=body, headers=upstream_headers())
        try:
            upstream = await client.send(req, stream=True)
        except httpx.TimeoutException:
            return _error(504, "Upstream provider timed out", "upstream_error", "upstream_timeout", rid)
        except httpx.HTTPError:
            logger.warning("rid=%s upstream connection failed", rid, exc_info=True)
            return _error(502, "Upstream provider unavailable", "upstream_error", "upstream_unavailable", rid)

        if upstream.status_code != 200:
            await upstream.aclose()  # body intentionally not relayed (may contain provider internals)
            logger.warning("rid=%s upstream status %s", rid, upstream.status_code)
            # 401/403 from the provider are the *gateway's* credential problem, not the client's: 502.
            status = upstream.status_code if upstream.status_code in (400, 404, 413, 422, 429) else 502
            return _error(
                status, "Upstream provider returned an error", "upstream_error", f"upstream_{upstream.status_code}", rid
            )

        if not stream:
            try:
                data = json.loads(await upstream.aread())
                count = redact_completion(data)
            except Exception:
                logger.exception("rid=%s failed to redact non-stream response", rid)
                return _error(502, "Response blocked by output guardrail", "guardrail_error", "guardrail_failure", rid)
            finally:
                await upstream.aclose()
            logger.info("rid=%s non-stream redactions=%d", rid, count)
            return JSONResponse(data, headers={"x-request-id": rid, "x-guardrail-redactions": str(count)})

        return StreamingResponse(
            with_heartbeat(_guarded_stream(upstream, rid, t0), settings.heartbeat_s),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no", "x-request-id": rid},
        )

    return app


async def _guarded_stream(upstream: httpx.Response, rid: str, t0: float) -> AsyncIterator[bytes]:
    decoder, guard = SSEDecoder(), ChunkGuard()
    first_out: float | None = None
    done = False
    try:
        async for raw in upstream.aiter_raw():
            for event in decoder.feed(raw):
                if event.data.strip() == "[DONE]":
                    done = True
                    break
                try:
                    chunk = json.loads(event.data)
                except ValueError:
                    continue  # not JSON (e.g. provider keep-alive data): nothing to forward
                if isinstance(chunk, dict) and "error" in chunk:
                    logger.warning("rid=%s upstream sent in-stream error", rid)
                    yield _error_event("Upstream provider error", "upstream_error", "upstream_stream_error", rid)
                    return
                out = guard.process(chunk)
                if out is not None:
                    if first_out is None:
                        first_out = time.perf_counter()
                    yield encode_event(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
            if done:
                break
        if not done:
            for _ in decoder.close():
                pass  # trailing partial event without [DONE]: treated as truncation below
        tail = guard.finish()
        if tail is not None:
            yield encode_event(json.dumps(tail, separators=(",", ":"), ensure_ascii=False))
        if not done:
            logger.warning("rid=%s upstream stream ended without [DONE]", rid)
            yield _error_event("Upstream stream ended unexpectedly", "upstream_error", "upstream_truncated", rid)
            return
        yield b"data: [DONE]\n\n"
    except httpx.ReadTimeout:
        logger.warning("rid=%s upstream stalled", rid)
        yield _error_event("Upstream provider stalled", "upstream_error", "upstream_timeout", rid)
    except httpx.HTTPError:
        logger.warning("rid=%s upstream stream broke", rid, exc_info=True)
        yield _error_event("Upstream connection lost", "upstream_error", "upstream_unavailable", rid)
    except Exception:
        # Fail closed: never fall back to forwarding raw upstream text.
        logger.exception("rid=%s guardrail failure; terminating stream", rid)
        yield _error_event("Response blocked by output guardrail", "guardrail_error", "guardrail_failure", rid)
    finally:
        await upstream.aclose()
        s = guard.stats
        logger.info(
            "rid=%s stream closed redactions=%d max_buffered_chars=%d chunks_in=%d chunks_out=%d ttft_ms=%s",
            rid,
            s.redactions,
            s.max_buffered,
            s.chunks_in,
            s.chunks_out,
            f"{(first_out - t0) * 1000:.1f}" if first_out else "n/a",
        )


def app_factory() -> FastAPI:
    logging.basicConfig(
        level=os.environ.get("GUARD_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    return create_app(GuardSettings.from_env())
