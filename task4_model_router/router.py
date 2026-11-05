"""Resilient model routing: hard per-attempt deadlines, failover, cooldown circuit breaker.

Timeout semantics (the subtle part)
-----------------------------------
httpx timeouts are *per phase* - ``read=3`` means "3 s between two socket reads", so a
provider that trickles a byte every 2 s never times out. The 3000 ms budget here is a
**total deadline** enforced with ``asyncio.timeout``: it covers connect, headers *and*
reading the body. When it fires, the in-flight request task is cancelled, httpx
closes that connection, and a late primary response can never be observed - so there
is no "both providers answered" race and no double-charging.

For ``stream=True`` the deadline is **time-to-first-token**: we don't commit to a
provider (or send a single byte to the client) until it has produced its first
content delta. A role-only delta doesn't count. Once committed, failover is
impossible (the client has already seen output), so later failures become a
sanitised in-stream error event.

Failover policy
---------------
fail over   429, 408, 5xx, 529 (overloaded), connect/read errors, deadline exceeded,
            401/402/403/404 (gateway-side credential/billing/model config for *that* provider)
don't       400, 413, 422: the request itself is bad; the secondary would reject it too.

Cooldown (circuit breaker, LiteLLM-style): a 429 parks the provider for its
``Retry-After`` (capped at 60 s, never slept on inside a request); timeouts/5xx park
it after ``failure_threshold`` consecutive failures. While parked, requests go straight
to the backup instead of each paying the 3 s timeout again.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from task3_llm_guardrail.sse import SSEDecoder

from .errors import GatewayError

logger = logging.getLogger("task4.router")

FAILOVER_STATUSES = frozenset({401, 402, 403, 404, 408, 409, 429, 500, 502, 503, 504, 529})


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key: str | None = None
    model: str | None = None  # override the requested model for this provider
    timeout_s: float = 3.0

    @property
    def url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


@dataclass
class Attempt:
    provider: str
    outcome: str  # ok | timeout | rate_limited | http_<status> | connection_error | bad_response | skipped_cooldown
    elapsed_ms: float


class AllProvidersFailed(Exception):
    def __init__(self, attempts: list[Attempt]) -> None:
        super().__init__("all providers failed")
        self.attempts = attempts


@dataclass
class _BreakerState:
    consecutive_failures: int = 0
    opened: bool = False
    open_until: float = 0.0
    probe_started: float | None = None  # half-open: a single probe request is in flight


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        cooldown_s: float = 30.0,
        rate_limit_default_s: float = 5.0,
        max_cooldown_s: float = 60.0,
        probe_timeout_s: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.probe_timeout_s = probe_timeout_s
        self.cooldown_s = cooldown_s
        self.rate_limit_default_s = rate_limit_default_s
        self.max_cooldown_s = max_cooldown_s
        self._clock = clock
        self._state: dict[str, _BreakerState] = {}

    def _s(self, name: str) -> _BreakerState:
        return self._state.setdefault(name, _BreakerState())

    def available(self, name: str) -> bool:
        """closed -> yes; open -> no; cooled down (half-open) -> yes for exactly one probe at a time,
        so a recovering provider gets one request, not a thundering herd that all pay the timeout."""
        s, now = self._s(name), self._clock()
        if not s.opened:
            return True
        if now < s.open_until:
            return False
        if s.probe_started is not None and now - s.probe_started < self.probe_timeout_s:
            return False
        s.probe_started = now
        return True

    def _open(self, s: _BreakerState, seconds: float) -> None:
        s.opened, s.open_until, s.probe_started, s.consecutive_failures = True, self._clock() + seconds, None, 0

    def record_success(self, name: str) -> None:
        self._state[name] = _BreakerState()

    def record_rate_limited(self, name: str, retry_after_s: float | None) -> None:
        wait = min(retry_after_s if retry_after_s is not None else self.rate_limit_default_s, self.max_cooldown_s)
        self._open(self._s(name), max(0.0, wait))

    def record_failure(self, name: str) -> None:
        s = self._s(name)
        if s.opened:  # failed half-open probe: straight back to open
            self._open(s, self.cooldown_s)
            return
        s.consecutive_failures += 1
        if s.consecutive_failures >= self.failure_threshold:
            self._open(s, self.cooldown_s)


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # HTTP-date form: fall back to the default cooldown


@dataclass
class RouteResult:
    body: dict[str, Any]
    provider: str
    fallback_used: bool
    attempts: list[Attempt]


@dataclass
class StreamRoute:
    provider: str
    fallback_used: bool
    attempts: list[Attempt]
    events: AsyncIterator[dict[str, Any] | str]  # decoded JSON chunks, "[DONE]", or {"error": ...}
    _close: Callable[[], Any] = field(repr=False, default=lambda: None)

    async def aclose(self) -> None:
        await self._close()


class ModelRouter:
    def __init__(
        self,
        providers: list[Provider],
        *,
        breaker: CircuitBreaker | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        stream_read_timeout_s: float = 30.0,
    ) -> None:
        if not providers:
            raise ValueError("at least one provider required")
        self.providers = providers
        self.breaker = breaker or CircuitBreaker()
        # httpx's own timeouts are a backstop only; asyncio.timeout() owns the per-attempt deadline.
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(connect=10.0, read=stream_read_timeout_s, write=10.0, pool=5.0),
            limits=httpx.Limits(max_connections=500, max_keepalive_connections=100),
            follow_redirects=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ helpers
    def _order(self, attempts: list[Attempt]) -> list[Provider]:
        ready = [p for p in self.providers if self.breaker.available(p.name)]
        for p in self.providers:
            if p not in ready:
                attempts.append(Attempt(p.name, "skipped_cooldown", 0.0))
        # Everyone cooling down: trying is better than failing without asking.
        return ready or list(self.providers)

    def _payload(self, provider: Provider, body: dict[str, Any]) -> dict[str, Any]:
        payload = dict(body)
        if provider.model:
            payload["model"] = provider.model
        return payload

    def _headers(self, provider: Provider, rid: str) -> dict[str, str]:
        h = {"content-type": "application/json", "x-request-id": rid}
        if provider.api_key:
            h["authorization"] = f"Bearer {provider.api_key}"
        return h

    def _classify(self, provider: Provider, resp: httpx.Response) -> str | None:
        """None -> usable success. Otherwise an outcome string, after updating the breaker.
        Raises GatewayError for non-failover client errors."""
        if resp.status_code == 429:
            self.breaker.record_rate_limited(provider.name, _retry_after_seconds(resp))
            return "rate_limited"
        if resp.status_code in FAILOVER_STATUSES or resp.status_code >= 500:
            self.breaker.record_failure(provider.name)
            return f"http_{resp.status_code}"
        if resp.status_code >= 400:
            # Bad request on our side - the backup would refuse it as well. Upstream body is NOT relayed.
            logger.info("provider %s rejected request with %s (not failing over)", provider.name, resp.status_code)
            raise GatewayError(
                400 if resp.status_code != 413 else 413,
                "invalid_request_error",
                "upstream_rejected_request",
                "The model provider rejected the request as invalid",
            )
        return None

    # ------------------------------------------------------------------ non-streaming
    async def complete(self, body: dict[str, Any], rid: str) -> RouteResult:
        attempts: list[Attempt] = []
        order = self._order(attempts)
        for provider in order:
            t0 = time.perf_counter()

            def done(outcome: str) -> None:
                attempts.append(Attempt(provider.name, outcome, round((time.perf_counter() - t0) * 1000, 1)))

            try:
                async with asyncio.timeout(provider.timeout_s):
                    resp = await self._client.post(
                        provider.url, json=self._payload(provider, body), headers=self._headers(provider, rid)
                    )  # .post() reads the full body - inside the deadline
            except TimeoutError:
                self.breaker.record_failure(provider.name)
                done("timeout")
                logger.warning("rid=%s provider=%s deadline %.0fms exceeded", rid, provider.name, provider.timeout_s * 1000)
                continue
            except httpx.HTTPError as exc:
                self.breaker.record_failure(provider.name)
                done("connection_error")
                logger.warning("rid=%s provider=%s transport error: %s", rid, provider.name, type(exc).__name__)
                continue
            except Exception:
                # Anything else from talking to this provider (decode errors, ...) is *its* failure:
                # fail over rather than surface a 500.
                self.breaker.record_failure(provider.name)
                done("bad_response")
                logger.exception("rid=%s provider=%s unexpected error", rid, provider.name)
                continue

            outcome = self._classify(provider, resp)
            if outcome is not None:
                done(outcome)
                logger.warning("rid=%s provider=%s failed: %s", rid, provider.name, outcome)
                continue
            try:
                data = resp.json()
                if not isinstance(data, dict) or not isinstance(data.get("choices"), list):
                    raise ValueError("unexpected response shape")
            except ValueError:
                self.breaker.record_failure(provider.name)
                done("bad_response")
                continue

            self.breaker.record_success(provider.name)
            done("ok")
            return RouteResult(data, provider.name, fallback_used=provider is not self.providers[0], attempts=attempts)
        raise AllProvidersFailed(attempts)

    # ------------------------------------------------------------------ streaming
    async def stream(self, body: dict[str, Any], rid: str) -> StreamRoute:
        attempts: list[Attempt] = []
        order = self._order(attempts)
        for provider in order:
            t0 = time.perf_counter()

            def done(outcome: str) -> None:
                attempts.append(Attempt(provider.name, outcome, round((time.perf_counter() - t0) * 1000, 1)))

            payload = self._payload(provider, body)
            payload["stream"] = True
            # Ask for the usage chunk so the limiter can reconcile actual tokens.
            payload["stream_options"] = {**(payload.get("stream_options") or {}), "include_usage": True}
            req = self._client.build_request("POST", provider.url, json=payload, headers=self._headers(provider, rid))
            resp: httpx.Response | None = None
            decoder = SSEDecoder()
            prefetched: list[dict[str, Any] | str] = []
            raw_iter = None
            try:
                async with asyncio.timeout(provider.timeout_s):  # deadline = time to first *content* token
                    resp = await self._client.send(req, stream=True)
                    outcome = self._classify(provider, resp)
                    if outcome is not None:
                        await resp.aclose()
                        done(outcome)
                        continue
                    raw_iter = resp.aiter_raw()
                    committed = False
                    while not committed:
                        chunk = await anext(raw_iter)
                        for ev in decoder.feed(chunk):
                            item = _decode_event(ev.data)
                            if item is None:
                                continue
                            if isinstance(item, dict) and "error" in item:
                                # Error before any output: nothing reached the client, so fail over.
                                raise _PreCommitStreamError
                            prefetched.append(item)
                            if _is_first_token(item):
                                committed = True
            except _PreCommitStreamError:
                await resp.aclose()  # type: ignore[union-attr]
                self.breaker.record_failure(provider.name)
                done("stream_error_before_first_token")
                continue
            except TimeoutError:
                if resp is not None:
                    await resp.aclose()
                self.breaker.record_failure(provider.name)
                done("timeout")
                logger.warning("rid=%s provider=%s no first token within %.0fms", rid, provider.name, provider.timeout_s * 1000)
                continue
            except (httpx.HTTPError, StopAsyncIteration):
                if resp is not None:
                    await resp.aclose()
                self.breaker.record_failure(provider.name)
                done("connection_error")
                continue
            except GatewayError:
                if resp is not None:
                    await resp.aclose()
                raise
            except Exception:
                # e.g. undecodable bytes before the first token: still nothing sent, still fail over.
                if resp is not None:
                    await resp.aclose()
                self.breaker.record_failure(provider.name)
                done("bad_response")
                logger.exception("rid=%s provider=%s unexpected stream error before first token", rid, provider.name)
                continue

            self.breaker.record_success(provider.name)
            done("ok")
            assert resp is not None and raw_iter is not None
            return StreamRoute(
                provider=provider.name,
                fallback_used=provider is not self.providers[0],
                attempts=attempts,
                events=_continue_stream(prefetched, raw_iter, decoder, rid, provider.name),
                _close=resp.aclose,
            )
        raise AllProvidersFailed(attempts)


class _PreCommitStreamError(Exception):
    pass


def _decode_event(data: str) -> dict[str, Any] | str | None:
    if data.strip() == "[DONE]":
        return "[DONE]"
    try:
        obj = json.loads(data)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _is_first_token(item: dict[str, Any] | str) -> bool:
    if item == "[DONE]":
        return True
    if not isinstance(item, dict):
        return False
    for choice in item.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("tool_calls") or delta.get("refusal") or choice.get("finish_reason"):
            return True
    return False


async def _continue_stream(prefetched, raw_iter, decoder, rid: str, provider: str):
    for item in prefetched:
        yield item
        if item == "[DONE]":
            return
    try:
        async for chunk in raw_iter:
            for ev in decoder.feed(chunk):
                item = _decode_event(ev.data)
                if item is None:
                    continue
                yield item
                if item == "[DONE]":
                    return
    except Exception:  # transport errors *and* decode errors (bad UTF-8, oversized event) after commit
        logger.warning("rid=%s provider=%s stream broke after first token", rid, provider, exc_info=True)
        yield {"error": {"message": "Upstream stream interrupted", "type": "upstream_error", "code": "upstream_stream_interrupted"}}
