"""Task 4 - LLM gateway: per-tenant token rate limiting + primary/backup model failover.

    POST /v1/chat/completions   (OpenAI-compatible; Authorization: Bearer <tenant API key>)

    1. authenticate tenant          sha256(key) looked up in SQLite `tenants`
    2. estimate + reserve tokens    SQLite sliding log, atomic (BEGIN IMMEDIATE)       → 429 + Retry-After
    3. route                        primary (3000 ms total deadline) → backup            → 503 if both fail
    4. reconcile                    reservation := usage.total_tokens (refund on failure)
    5. respond                      x-ratelimit-*, x-gateway-provider, x-gateway-fallback, x-request-id

Run:
    ROUTER_DB_PATH=./data/router.db \
    ROUTER_TENANTS="tenant-key-1:acme,tenant-key-2:globex:100000" \
    ROUTER_PRIMARY_URL=http://127.0.0.1:8401/v1 ROUTER_SECONDARY_URL=http://127.0.0.1:8402/v1 \
    uv run uvicorn --factory task4_model_router.app:app_factory --port 8400
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import GatewayError, error_body, install_error_handlers
from .limiter import RateLimited, RequestTooLarge, SQLiteTokenLimiter, WindowState
from .router import AllProvidersFailed, CircuitBreaker, ModelRouter, Provider
from .tokens import estimate_prompt_tokens, estimate_text_tokens

logger = logging.getLogger("task4.app")


# --------------------------------------------------------------------------- request model
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # provider-specific params pass through
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1, max_length=2048)
    max_tokens: int | None = Field(default=None, ge=1, le=200_000)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=200_000)
    stream: bool = False
    n: int | None = Field(default=None, ge=1, le=128)
    tools: list[dict[str, Any]] | None = None


def _output_chars(node: Any) -> int:
    """Characters of model output in a choice/delta - every string, not just ``content`` (tool-call
    arguments and reasoning are billed output too)."""
    if isinstance(node, str):
        return len(node)
    if isinstance(node, dict):
        return sum(_output_chars(v) for k, v in node.items() if k not in ("role", "type", "id", "index", "finish_reason"))
    if isinstance(node, list):
        return sum(_output_chars(v) for v in node)
    return 0


# --------------------------------------------------------------------------- settings
@dataclass
class RouterSettings:
    db_path: str = "data/router.db"
    tokens_per_minute: int = 50_000
    window_ms: int = 60_000
    default_max_tokens: int = 1024
    primary: Provider = field(default_factory=lambda: Provider("primary", "http://127.0.0.1:8401/v1", timeout_s=3.0))
    secondary: Provider = field(default_factory=lambda: Provider("secondary", "http://127.0.0.1:8402/v1", timeout_s=10.0))
    tenants: list[tuple[str, str, int | None]] = field(default_factory=list)  # (api_key, name, tpm override)
    prune_interval_s: float = 30.0
    max_body_bytes: int = 4_000_000

    @classmethod
    def from_env(cls) -> RouterSettings:
        tenants = []
        for spec in filter(None, os.environ.get("ROUTER_TENANTS", "").split(",")):
            parts = spec.split(":")
            tenants.append((parts[0], parts[1] if len(parts) > 1 else parts[0][:8], int(parts[2]) if len(parts) > 2 else None))
        return cls(
            db_path=os.environ.get("ROUTER_DB_PATH", cls.db_path),
            tokens_per_minute=int(os.environ.get("ROUTER_TPM", 50_000)),
            primary=Provider(
                "primary",
                os.environ.get("ROUTER_PRIMARY_URL", "http://127.0.0.1:8401/v1"),
                api_key=os.environ.get("ROUTER_PRIMARY_API_KEY"),
                model=os.environ.get("ROUTER_PRIMARY_MODEL"),
                timeout_s=float(os.environ.get("ROUTER_PRIMARY_TIMEOUT_MS", 3000)) / 1000,
            ),
            secondary=Provider(
                "secondary",
                os.environ.get("ROUTER_SECONDARY_URL", "http://127.0.0.1:8402/v1"),
                api_key=os.environ.get("ROUTER_SECONDARY_API_KEY"),
                model=os.environ.get("ROUTER_SECONDARY_MODEL"),
                timeout_s=float(os.environ.get("ROUTER_SECONDARY_TIMEOUT_MS", 10000)) / 1000,
            ),
            tenants=tenants,
        )


def _fmt_duration(ms: int) -> str:
    """OpenAI's x-ratelimit-reset-* format: 1m30s, 6s, 250ms."""
    if ms < 1000:
        return f"{ms}ms"
    m, s = divmod(round(ms / 1000), 60)
    return f"{m}m{s}s" if m else f"{s}s"


def ratelimit_headers(state: WindowState) -> dict[str, str]:
    return {
        "x-ratelimit-limit-tokens": str(state.limit),
        "x-ratelimit-remaining-tokens": str(state.remaining),
        "x-ratelimit-reset-tokens": _fmt_duration(state.reset_ms),
    }


# --------------------------------------------------------------------------- app
class RequestIdMiddleware:
    """Pure ASGI (not BaseHTTPMiddleware, which buffers/meddles with streaming and disconnects)."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            scope.setdefault("state", {})["request_id"] = f"req_{uuid.uuid4().hex}"
        await self.app(scope, receive, send)


def create_app(
    settings: RouterSettings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    clock=time.time,
    breaker: CircuitBreaker | None = None,
) -> FastAPI:
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    limiter = SQLiteTokenLimiter(
        settings.db_path, tokens_per_window=settings.tokens_per_minute, window_ms=settings.window_ms, clock=clock
    )
    for api_key, name, tpm in settings.tenants:
        limiter.add_tenant(api_key, name, tpm)
    router = ModelRouter([settings.primary, settings.secondary], transport=transport, breaker=breaker)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        async def pruner():
            while True:
                await asyncio.sleep(settings.prune_interval_s)
                with contextlib.suppress(Exception):
                    if n := await limiter.prune():
                        logger.info("pruned %d expired usage rows", n)

        task = asyncio.create_task(pruner())
        yield
        task.cancel()
        await router.aclose()
        limiter.close()

    app = FastAPI(title="LLM Gateway - Rate Limit & Fallback Router", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.limiter, app.state.router = limiter, router
    install_error_handlers(app)

    app.add_middleware(RequestIdMiddleware)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    async def authenticate(request: Request) -> tuple[str, str, int]:
        auth = request.headers.get("authorization", "")
        scheme, _, key = auth.partition(" ")
        if scheme.lower() != "bearer" or not key.strip():
            raise GatewayError(401, "authentication_error", "missing_api_key", "Provide your API key as 'Authorization: Bearer <key>'",
                               headers={"www-authenticate": "Bearer"})
        tenant = await asyncio.to_thread(limiter.tenant, key.strip())
        if tenant is None:
            raise GatewayError(401, "authentication_error", "invalid_api_key", "Invalid API key", headers={"www-authenticate": "Bearer"})
        return tenant

    async def read_body(request: Request) -> bytes:
        declared = request.headers.get("content-length")
        too_large = GatewayError(413, "invalid_request_error", "request_too_large",
                                 f"Request body exceeds {settings.max_body_bytes} bytes")
        if declared and declared.isdigit() and int(declared) > settings.max_body_bytes:
            raise too_large
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > settings.max_body_bytes:
                raise too_large
            chunks.append(chunk)
        return b"".join(chunks)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        rid: str = request.state.request_id
        # Authenticate *before* touching the body: unauthenticated callers can't make us buffer and
        # parse large payloads, nor learn our schema from validation errors.
        key_hash, tenant_name, limit = await authenticate(request)
        raw = await read_body(request)
        try:
            body = ChatRequest.model_validate_json(raw)
        except ValidationError as exc:
            raise RequestValidationError(exc.errors(include_url=False)) from None
        payload = body.model_dump(exclude_none=True)

        prompt_est = estimate_prompt_tokens(payload["messages"], payload.get("tools"))
        completion_budget = body.max_completion_tokens or body.max_tokens or settings.default_max_tokens
        # n choices each generate up to max_tokens: reserve for all of them.
        reserve_tokens = prompt_est + (body.n or 1) * completion_budget
        try:
            decision = await limiter.reserve(key_hash, reserve_tokens, limit, rid)
        except RequestTooLarge as exc:
            # Non-retryable: no amount of waiting makes this fit, so don't return 429.
            raise GatewayError(
                400, "invalid_request_error", "request_exceeds_token_limit",
                f"This request needs ~{exc.tokens} tokens (prompt + max_tokens) but your limit is {exc.limit} per minute. "
                "Reduce max_tokens or the prompt size.",
                param="max_tokens",
            ) from None
        if isinstance(decision, RateLimited):
            logger.info("rid=%s tenant=%s rate limited (needs %d, used %d/%d)", rid, tenant_name, reserve_tokens,
                        decision.state.used, limit)
            raise GatewayError(
                429, "rate_limit_error", "tenant_token_limit_exceeded",
                f"Rate limit reached: {limit} tokens per minute. Retry after {decision.retry_after_s}s.",
                headers={"retry-after": str(decision.retry_after_s), **ratelimit_headers(decision.state)},
            )
        reservation = decision

        def final_headers(actual: int, provider: str, fallback: bool) -> dict[str, str]:
            state = reservation.state
            adjusted = WindowState(state.limit, state.used - reservation.tokens + actual, state.reset_ms)
            return {
                "x-request-id": rid,
                "x-gateway-provider": provider,
                "x-gateway-fallback": "true" if fallback else "false",
                **ratelimit_headers(adjusted),
            }

        async def unavailable(exc: AllProvidersFailed) -> GatewayError:
            await limiter.release(reservation.id)  # nothing was generated: refund
            logger.error("rid=%s all providers failed: %s", rid, [(a.provider, a.outcome, a.elapsed_ms) for a in exc.attempts])
            return GatewayError(503, "upstream_error", "all_providers_unavailable",
                                "All model providers are currently unavailable. Please retry shortly.",
                                headers={"retry-after": "5"})

        # ---------------------------------------------------------- non-streaming
        if not body.stream:
            try:
                result = await router.complete(payload, rid)
            except AllProvidersFailed as exc:
                raise await unavailable(exc) from None
            except BaseException:  # GatewayError, unexpected bugs, cancellation: nothing delivered -> refund
                await asyncio.shield(limiter.release(reservation.id))
                raise
            usage = result.body.get("usage") or {}
            actual = usage.get("total_tokens")
            if not isinstance(actual, int) or actual < 0:
                actual = prompt_est + estimate_text_tokens(" " * _output_chars(result.body.get("choices", [])))
            await limiter.reconcile(reservation.id, actual, reservation.key_hash)
            logger.info("rid=%s tenant=%s provider=%s fallback=%s reserved=%d actual=%d attempts=%s", rid, tenant_name,
                        result.provider, result.fallback_used, reservation.tokens, actual,
                        [(a.provider, a.outcome) for a in result.attempts])
            return JSONResponse(result.body, headers=final_headers(actual, result.provider, result.fallback_used))

        # ---------------------------------------------------------- streaming
        client_wants_usage = bool((payload.get("stream_options") or {}).get("include_usage"))
        try:
            route = await router.stream(payload, rid)
        except AllProvidersFailed as exc:
            raise await unavailable(exc) from None
        except BaseException:
            await asyncio.shield(limiter.release(reservation.id))
            raise

        async def relay():
            delivered_chars = 0
            actual: int | None = None
            try:
                async for item in route.events:
                    if item == "[DONE]":
                        yield b"data: [DONE]\n\n"
                        break
                    assert isinstance(item, dict)
                    if "error" in item:
                        # Committed already - no failover possible. Sanitise and end.
                        yield _sse(error_body("upstream_error", "upstream_stream_interrupted",
                                              "The model provider stream was interrupted", rid))
                        break
                    if isinstance(item.get("usage"), dict) and isinstance(item["usage"].get("total_tokens"), int):
                        actual = item["usage"]["total_tokens"]
                        if not client_wants_usage and not item.get("choices"):
                            continue  # we asked for the usage chunk for accounting; client didn't
                    delivered_chars += _output_chars(item.get("choices") or [])
                    yield _sse(item)
            finally:
                await route.aclose()
                # Client disconnects still pay for what was generated; refund only if nothing was.
                charge = actual if actual is not None else prompt_est + estimate_text_tokens(" " * delivered_chars)
                await asyncio.shield(limiter.reconcile(reservation.id, charge, reservation.key_hash))
                logger.info("rid=%s tenant=%s stream provider=%s fallback=%s charged=%d", rid, tenant_name,
                            route.provider, route.fallback_used, charge)

        headers = final_headers(reservation.tokens, route.provider, route.fallback_used)
        headers.update({"cache-control": "no-cache", "x-accel-buffering": "no"})
        return StreamingResponse(relay(), media_type="text/event-stream", headers=headers)

    return app


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n".encode()


def app_factory() -> FastAPI:
    logging.basicConfig(level=os.environ.get("ROUTER_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return create_app(RouterSettings.from_env())
