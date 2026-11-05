"""Task 4: token-aware SQLite sliding-window limiter + failover router."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from task4_model_router.app import RouterSettings, create_app
from task4_model_router.limiter import RateLimited, RequestTooLarge, Reservation, SQLiteTokenLimiter, hash_key
from task4_model_router.mock_providers import build as build_provider
from task4_model_router.router import CircuitBreaker, Provider

from .conftest import run_uvicorn

ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


# =============================================================================== limiter
@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def limiter(tmp_path, clock):
    lim = SQLiteTokenLimiter(tmp_path / "rl.db", tokens_per_window=50_000, window_ms=60_000, clock=clock)
    yield lim
    lim.close()


def test_admits_up_to_limit_then_rejects(limiter):
    k = hash_key("tenant-a")
    for _ in range(5):
        assert isinstance(limiter.reserve_sync(k, 10_000), Reservation)
    denied = limiter.reserve_sync(k, 1)
    assert isinstance(denied, RateLimited)
    assert denied.state.used == 50_000 and denied.state.remaining == 0


def test_sliding_window_expiry_is_exact(limiter, clock):
    k = hash_key("t")
    limiter.reserve_sync(k, 30_000)  # t=0
    clock.advance(30)
    limiter.reserve_sync(k, 20_000)  # t=30
    assert isinstance(limiter.reserve_sync(k, 1), RateLimited)
    clock.advance(29.999)  # t=59.999: first entry still inside the window
    assert isinstance(limiter.reserve_sync(k, 1), RateLimited)
    clock.advance(0.001)  # t=60.000: first entry leaves the window exactly now
    assert isinstance(limiter.reserve_sync(k, 30_000), Reservation)
    assert isinstance(limiter.reserve_sync(k, 1), RateLimited)


def test_no_fixed_window_boundary_burst(limiter, clock):
    """A fixed-window limiter would allow 2x the limit across a boundary; a sliding log doesn't."""
    k = hash_key("t")
    clock.advance(59)
    limiter.reserve_sync(k, 50_000)
    clock.advance(2)  # a fixed minute window would have reset here
    assert isinstance(limiter.reserve_sync(k, 1_000), RateLimited)


def test_retry_after_walks_oldest_first(limiter, clock):
    k = hash_key("t")
    limiter.reserve_sync(k, 10_000)  # t=0  expires at 60
    clock.advance(10)
    limiter.reserve_sync(k, 10_000)  # t=10 expires at 70
    clock.advance(10)
    limiter.reserve_sync(k, 30_000)  # t=20 expires at 80
    clock.advance(5)  # now t=25, used=50k
    # Needs 15k: freeing the t=0 entry (10k) isn't enough; t=0 + t=10 (20k) is -> retry at t=70 -> 45s
    denied = limiter.reserve_sync(k, 15_000)
    assert isinstance(denied, RateLimited)
    assert denied.retry_after_ms == 45_000 and denied.retry_after_s == 45
    clock.advance(44.999)
    assert isinstance(limiter.reserve_sync(k, 15_000), RateLimited)
    clock.advance(0.001)
    assert isinstance(limiter.reserve_sync(k, 15_000), Reservation)


def test_reconcile_returns_capacity_and_release_refunds(limiter):
    k = hash_key("t")
    r = limiter.reserve_sync(k, 40_000)
    assert isinstance(limiter.reserve_sync(k, 20_000), RateLimited)
    limiter.reconcile_sync(r.id, 1_500)  # provider said the call actually used 1.5k tokens
    assert isinstance(limiter.reserve_sync(k, 20_000), Reservation)
    assert limiter.state_sync(k).used == 21_500
    limiter.release_sync(r.id)
    assert limiter.state_sync(k).used == 20_000


def test_request_bigger_than_limit_is_non_retryable(limiter):
    with pytest.raises(RequestTooLarge):
        limiter.reserve_sync(hash_key("t"), 50_001)


def test_tenants_isolated_and_custom_limits(limiter):
    limiter.reserve_sync(hash_key("a"), 50_000)
    assert isinstance(limiter.reserve_sync(hash_key("b"), 50_000), Reservation)
    limiter.add_tenant("big-key", "big", tpm_limit=200_000)
    kh, name, lim = limiter.tenant("big-key")
    assert (name, lim) == ("big", 200_000)
    assert isinstance(limiter.reserve_sync(kh, 150_000, lim), Reservation)


def test_eviction_deletes_rows(limiter, clock):
    for i in range(50):
        limiter.reserve_sync(hash_key(f"t{i}"), 100)
    assert limiter.row_count() == 50
    clock.advance(61)
    limiter.state_sync(hash_key("t0"))  # per-key eviction inside the read transaction
    assert limiter.row_count() == 49
    assert limiter.prune_sync() == 49  # periodic sweep for idle keys
    assert limiter.row_count() == 0


def test_state_persists_on_disk(tmp_path, clock):
    path = tmp_path / "persist.db"
    a = SQLiteTokenLimiter(path, clock=clock)
    a.reserve_sync(hash_key("t"), 45_000)
    a.close()
    b = SQLiteTokenLimiter(path, clock=clock)  # e.g. after a gateway restart
    assert isinstance(b.reserve_sync(hash_key("t"), 10_000), RateLimited)
    b.close()


def test_api_keys_never_stored_in_plaintext(limiter):
    limiter.add_tenant("sk-tenant-plaintext-key", "acme")
    raw = Path(limiter.db_path).read_bytes() + Path(limiter.db_path + "-wal").read_bytes()
    assert b"sk-tenant-plaintext-key" not in raw


def test_thread_concurrency_never_overshoots(tmp_path):
    lim = SQLiteTokenLimiter(tmp_path / "c.db", tokens_per_window=50_000)
    k = hash_key("t")
    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(lambda _: lim.reserve_sync(k, 2_000), range(200)))
    admitted = sum(isinstance(r, Reservation) for r in results)
    assert admitted == 25
    assert lim.state_sync(k).used == 50_000
    lim.close()


def test_multi_process_concurrency_never_overshoots(tmp_path):
    """Several gateway workers sharing one DB file: BEGIN IMMEDIATE makes check-and-reserve atomic."""
    db = tmp_path / "mp.db"
    SQLiteTokenLimiter(db).close()
    worker = textwrap.dedent(f"""
        import sys
        from task4_model_router.limiter import SQLiteTokenLimiter, Reservation, hash_key
        lim = SQLiteTokenLimiter({str(db)!r}, tokens_per_window=50_000)
        ok = sum(isinstance(lim.reserve_sync(hash_key("shared"), 1_000), Reservation) for _ in range(40))
        print(ok)
    """)
    procs = [
        subprocess.Popen([sys.executable, "-c", worker], cwd=ROOT, stdout=subprocess.PIPE, text=True) for _ in range(4)
    ]
    admitted = sum(int(p.communicate(timeout=60)[0].strip()) for p in procs)
    assert admitted == 50  # 4 x 40 attempts of 1k tokens against a 50k budget
    lim = SQLiteTokenLimiter(db)
    assert lim.state_sync(hash_key("shared")).used == 50_000
    lim.close()


# =============================================================================== router / app
TENANT = "tenant-key-acme"


@pytest.fixture(scope="module")
def providers():
    p_app, p_state = build_provider("primary")
    s_app, s_state = build_provider("secondary")
    with run_uvicorn(p_app) as p_url, run_uvicorn(s_app) as s_url:
        yield {"p_url": p_url, "s_url": s_url, "p": p_state, "s": s_state}


@pytest.fixture
def gw(providers, tmp_path):
    for st in (providers["p"], providers["s"]):
        st.update(mode="ok", delay_s=0.0, retry_after="2", calls=0)
    settings = RouterSettings(
        db_path=str(tmp_path / "gw.db"),
        primary=Provider("primary", f"{providers['p_url']}/v1", timeout_s=3.0),
        secondary=Provider("secondary", f"{providers['s_url']}/v1", timeout_s=5.0),
        tenants=[(TENANT, "acme", None), ("tenant-key-tiny", "tiny", 3_000)],
    )
    with run_uvicorn(create_app(settings)) as url:
        yield url


def chat(url: str, key: str = TENANT, **body) -> httpx.Response:
    payload = {"model": "gpt-x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100, **body}
    return httpx.post(
        f"{url}/v1/chat/completions", json=payload, headers={"authorization": f"Bearer {key}"}, timeout=30
    )


def assert_error_envelope(resp: httpx.Response, status: int, err_type: str, code: str) -> dict:
    assert resp.status_code == status, resp.text
    err = resp.json()["error"]
    assert set(err) == {"message", "type", "code", "param", "request_id"}
    assert (err["type"], err["code"]) == (err_type, code)
    assert err["request_id"] == resp.headers["x-request-id"]
    for leak in ("Traceback", "shard_router", "org-SECRET", "sk-live", "/opt/provider", "httpx", "Exception"):
        assert leak not in resp.text
    return err


def test_primary_success(gw, providers):
    r = chat(gw)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "Hello from primary"
    assert r.headers["x-gateway-provider"] == "primary" and r.headers["x-gateway-fallback"] == "false"
    # reserved ~ 100 + prompt, reconciled to the provider's actual usage (25)
    assert r.headers["x-ratelimit-limit-tokens"] == "50000"
    assert r.headers["x-ratelimit-remaining-tokens"] == str(50_000 - 25)


def test_failover_on_429(gw, providers):
    providers["p"]["mode"] = "rate_limited"
    r = chat(gw)
    assert r.status_code == 200 and r.headers["x-gateway-provider"] == "secondary"
    assert r.headers["x-gateway-fallback"] == "true"


def test_failover_on_timeout_at_3000ms(gw, providers):
    providers["p"].update(mode="slow", delay_s=4.0)
    t0 = time.perf_counter()
    r = chat(gw)
    elapsed = time.perf_counter() - t0
    assert r.status_code == 200 and r.headers["x-gateway-provider"] == "secondary"
    assert 3.0 <= elapsed < 3.8, f"failover should happen at the 3 s deadline, took {elapsed:.2f}s"


def test_total_deadline_not_per_read_timeout(gw, providers):
    """Primary sends headers instantly then trickles a byte every 0.5 s: a per-read timeout never fires."""
    providers["p"]["mode"] = "trickle"
    t0 = time.perf_counter()
    r = chat(gw)
    assert r.headers["x-gateway-provider"] == "secondary"
    assert time.perf_counter() - t0 < 3.8


@pytest.mark.parametrize("mode", ["server_error"])
def test_failover_on_5xx(gw, providers, mode):
    providers["p"]["mode"] = mode
    assert chat(gw).headers["x-gateway-provider"] == "secondary"


def test_failover_on_connection_refused(providers, tmp_path):
    settings = RouterSettings(
        db_path=str(tmp_path / "x.db"),
        primary=Provider("primary", "http://127.0.0.1:1/v1"),
        secondary=Provider("secondary", f"{providers['s_url']}/v1"),
        tenants=[(TENANT, "acme", None)],
    )
    providers["s"]["mode"] = "ok"
    with run_uvicorn(create_app(settings)) as url:
        assert chat(url).headers["x-gateway-provider"] == "secondary"


def test_no_failover_on_400_and_body_not_leaked(gw, providers):
    providers["p"]["mode"] = "bad_request"
    r = chat(gw)
    assert_error_envelope(r, 400, "invalid_request_error", "upstream_rejected_request")
    assert providers["s"]["calls"] == 0


def test_all_providers_down_sanitised_and_refunded(gw, providers):
    providers["p"]["mode"] = "server_error"
    providers["s"]["mode"] = "rate_limited"
    r = chat(gw, max_tokens=40_000)
    assert_error_envelope(r, 503, "upstream_error", "all_providers_unavailable")
    providers["p"]["mode"] = providers["s"]["mode"] = "ok"
    # the 40k reservation was refunded, so a second 40k request still fits in the 50k window
    assert chat(gw, max_tokens=40_000).status_code == 200


def test_cooldown_skips_rate_limited_primary(gw, providers):
    providers["p"].update(mode="rate_limited", retry_after="30")
    assert chat(gw).headers["x-gateway-provider"] == "secondary"
    calls = providers["p"]["calls"]
    providers["p"]["mode"] = "ok"  # primary recovered, but its Retry-After hasn't elapsed
    for _ in range(3):
        assert chat(gw).headers["x-gateway-provider"] == "secondary"
    assert providers["p"]["calls"] == calls, "primary must not be hit during its cooldown"


def test_circuit_breaker_timing():
    clock = FakeClock(0)
    cb = CircuitBreaker(failure_threshold=2, cooldown_s=30, clock=clock)
    cb.record_failure("p")
    assert cb.available("p")
    cb.record_failure("p")
    assert not cb.available("p")
    clock.advance(30)
    assert cb.available("p")  # half-open: next request probes it
    cb.record_rate_limited("p", 600)  # absurd Retry-After is capped
    clock.advance(60)
    assert cb.available("p")


def test_rate_limit_429_envelope_and_headers(gw):
    ok = chat(gw, key="tenant-key-tiny", max_tokens=2_000)
    assert ok.status_code == 200
    # tiny tenant: 3k/min. Actual usage was 25, so 2.9k more fits; 2.98k+prompt doesn't
    r = chat(gw, key="tenant-key-tiny", max_tokens=2_980)
    err = assert_error_envelope(r, 429, "rate_limit_error", "tenant_token_limit_exceeded")
    assert int(r.headers["retry-after"]) >= 1
    assert r.headers["x-ratelimit-limit-tokens"] == "3000"
    assert "Retry after" in err["message"]


def test_request_exceeding_limit_is_400_not_429(gw):
    r = chat(gw, max_tokens=60_000)
    assert_error_envelope(r, 400, "invalid_request_error", "request_exceeds_token_limit")
    assert "retry-after" not in r.headers


def test_auth_errors(gw):
    assert_error_envelope(chat(gw, key="nope"), 401, "authentication_error", "invalid_api_key")
    r = httpx.post(f"{gw}/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
    assert_error_envelope(r, 401, "authentication_error", "missing_api_key")


@pytest.mark.parametrize(
    "body",
    [
        {"messages": []},
        {"messages": [{"role": "wizard", "content": "x"}]},
        {"messages": "hi"},
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": -5},
    ],
)
def test_validation_errors_standardised(gw, body):
    r = httpx.post(f"{gw}/v1/chat/completions", json=body, headers={"authorization": f"Bearer {TENANT}"})
    assert_error_envelope(r, 400, "invalid_request_error", "invalid_request")


def test_malformed_json_and_unknown_route(gw):
    r = httpx.post(
        f"{gw}/v1/chat/completions",
        content=b"{oops",
        headers={"authorization": f"Bearer {TENANT}", "content-type": "application/json"},
    )
    assert_error_envelope(r, 400, "invalid_request_error", "invalid_request")
    assert_error_envelope(httpx.get(f"{gw}/admin/secrets"), 404, "invalid_request_error", "not_found")


def test_concurrent_requests_respect_limit(gw, providers):
    """30 concurrent requests reserving ~10k each against 50k: at most 5 can be in flight at once."""
    providers["p"].update(mode="slow", delay_s=0.5)
    with ThreadPoolExecutor(max_workers=30) as pool:
        codes = list(pool.map(lambda _: chat(gw, max_tokens=9_990).status_code, range(30)))
    assert codes.count(200) == 5 and codes.count(429) == 25


# =============================================================================== streaming
def _sse_items(text: str) -> list:
    return [json.loads(l[6:]) if l[6:] != "[DONE]" else "[DONE]" for l in text.splitlines() if l.startswith("data: ")]


def test_stream_primary(gw, providers):
    r = chat(gw, stream=True)
    items = _sse_items(r.text)
    assert r.headers["x-gateway-provider"] == "primary" and items[-1] == "[DONE]"
    content = "".join(c["delta"].get("content", "") for i in items if isinstance(i, dict) for c in i.get("choices", []))
    assert content.strip() == "Hello from primary"
    assert all(not (isinstance(i, dict) and "usage" in i) for i in items), "usage chunk stripped unless requested"
    assert providers["p"]["last_body"]["stream_options"] == {"include_usage": True}


def test_stream_ttft_timeout_fails_over(gw, providers):
    providers["p"].update(mode="stream_slow_first", delay_s=4.0)  # role chunk at once, no content for 4 s
    t0 = time.perf_counter()
    r = chat(gw, stream=True)
    assert r.headers["x-gateway-provider"] == "secondary" and r.headers["x-gateway-fallback"] == "true"
    assert time.perf_counter() - t0 < 3.8
    assert "Hello from secondary" in "".join(
        c["delta"].get("content", "") for i in _sse_items(r.text) if isinstance(i, dict) for c in i.get("choices", [])
    )


def test_stream_break_after_commit_is_sanitised_in_stream(gw, providers):
    providers["p"]["mode"] = "stream_break"
    r = chat(gw, stream=True)
    assert r.status_code == 200 and r.headers["x-gateway-provider"] == "primary"
    last = _sse_items(r.text)[-1]
    assert last["error"]["code"] == "upstream_stream_interrupted"
    assert providers["s"]["calls"] == 0  # already committed: no failover after first token


def test_stream_usage_passed_through_when_requested(gw):
    items = _sse_items(chat(gw, stream=True, stream_options={"include_usage": True}).text)
    assert any(isinstance(i, dict) and i.get("usage", {}).get("total_tokens") == 25 for i in items)


# =============================================================================== iteration 3 (adversarial review regressions)
from task4_model_router.router import ModelRouter  # noqa: E402


@pytest.mark.parametrize("mode", ["stream_bad_bytes", "stream_error_first"])
def test_stream_failure_before_first_token_fails_over(gw, providers, mode):
    providers["p"]["mode"] = mode
    r = chat(gw, stream=True)
    assert r.status_code == 200 and r.headers["x-gateway-provider"] == "secondary"
    assert "shard" not in r.text
    assert providers["s"]["calls"] == 1


def test_reservation_accounts_for_n(gw):
    # 13 choices x 4000 max_tokens can never fit in 50k: must be counted, not reserved as 4000.
    r = chat(gw, n=13, max_tokens=4_000)
    assert_error_envelope(r, 400, "invalid_request_error", "request_exceeds_token_limit")


def test_auth_happens_before_body_parsing(gw):
    url = f"{gw}/v1/chat/completions"
    # schema-invalid but unauthenticated -> 401, not a schema-revealing 400
    assert httpx.post(url, json={"messages": "x"}).status_code == 401
    # large unauthenticated body is refused without being parsed
    big = b'{"messages":[{"role":"user","content":"' + b"a" * 3_000_000 + b'"}]}'
    t0 = time.perf_counter()
    r = httpx.post(url, content=big, headers={"content-type": "application/json"})
    assert r.status_code == 401 and time.perf_counter() - t0 < 2


def test_body_size_limit_standardised(gw):
    big = b'{"messages":[{"role":"user","content":"' + b"a" * 5_000_000 + b'"}]}'
    r = httpx.post(
        f"{gw}/v1/chat/completions",
        content=big,
        headers={"authorization": f"Bearer {TENANT}", "content-type": "application/json"},
    )
    assert_error_envelope(r, 413, "invalid_request_error", "request_too_large")


def test_list_content_without_usage_is_charged_not_500(gw, providers):
    providers["p"]["mode"] = "list_no_usage"
    r = chat(gw)
    assert r.status_code == 200
    assert int(r.headers["x-ratelimit-remaining-tokens"]) < 50_000  # charged an estimate, not free


def test_reconcile_after_eviction_reinserts_usage(limiter, clock):
    k = hash_key("long-stream")
    r = limiter.reserve_sync(k, 1_000)
    clock.advance(61)  # stream outlived the window; the reservation row gets evicted
    limiter.state_sync(k)
    limiter.reconcile_sync(r.id, 30_000, k)
    assert limiter.state_sync(k).used == 30_000  # consumption is not silently forgiven


def test_half_open_allows_single_probe():
    clock = FakeClock(0)
    cb = CircuitBreaker(failure_threshold=1, cooldown_s=10, probe_timeout_s=5, clock=clock)
    cb.record_failure("p")
    clock.advance(10)
    assert cb.available("p") is True  # first caller gets the probe
    assert cb.available("p") is False  # concurrent callers don't pile onto a maybe-dead provider
    cb.record_failure("p")  # probe failed -> open again immediately
    assert cb.available("p") is False
    clock.advance(10)
    assert cb.available("p") is True
    cb.record_success("p")
    assert cb.available("p") and cb.available("p")  # closed: everyone


async def test_fallback_flag_reflects_serving_provider():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"index": 0, "message": {"content": "hi"}}], "usage": {"total_tokens": 3}}
        )

    clock = FakeClock(0)
    breaker = CircuitBreaker(clock=clock, rate_limit_default_s=5)
    router = ModelRouter(
        [Provider("primary", "http://p/v1"), Provider("secondary", "http://s/v1")],
        breaker=breaker,
        transport=httpx.MockTransport(handler),
    )
    breaker.record_rate_limited("primary", 5)
    breaker.record_rate_limited("secondary", 5)  # everyone cooling down -> primary still tried first
    result = await router.complete({"messages": []}, "rid")
    await router.aclose()
    assert result.provider == "primary" and result.fallback_used is False


# =============================================================================== iteration 4 (peer benchmark)
def test_mock_providers_controllable_at_runtime(providers):
    """Reviewers can trigger failover by hand: POST /_mock/mode, GET /_mock/stats, or x-mock-mode header."""
    r = httpx.post(f"{providers['p_url']}/_mock/mode", json={"mode": "rate_limited", "retry_after": "7"})
    assert r.json()["mode"] == "rate_limited"
    resp = httpx.post(f"{providers['p_url']}/v1/chat/completions", json={"messages": []})
    assert resp.status_code == 429 and resp.headers["retry-after"] == "7"
    httpx.post(f"{providers['p_url']}/_mock/mode", json={"mode": "ok"})
    resp = httpx.post(
        f"{providers['p_url']}/v1/chat/completions", json={"messages": []}, headers={"x-mock-mode": "server_error"}
    )
    assert resp.status_code == 500
    assert httpx.get(f"{providers['p_url']}/_mock/stats").json()["calls"] >= 2


def _composed(providers, tmp_path, **kw):
    settings = RouterSettings(
        db_path=str(tmp_path / "c.db"),
        primary=Provider("primary", f"{providers['p_url']}/v1"),
        secondary=Provider("secondary", f"{providers['s_url']}/v1"),
        tenants=[(TENANT, "acme", None)],
        **kw,
    )
    return create_app(settings)


def test_composed_router_redacts_pii(providers, tmp_path):
    """Task 3 inside Task 4: one endpoint with rate limiting, failover *and* streaming PII redaction."""
    for st in (providers["p"], providers["s"]):
        st.update(mode="ok", calls=0)
    providers["p"]["mode"] = "rate_limited"  # force failover too
    with run_uvicorn(_composed(providers, tmp_path, redact_pii=True)) as url:
        # the mock echoes "Hello from <name>": use a name with PII to prove redaction on both paths
        providers["s"]["name"] = "bob@corp.example"
        r = chat(url)
        s = chat(url, stream=True)
    providers["s"]["name"] = "secondary"
    assert r.headers["x-gateway-provider"] == "secondary"
    assert "bob@corp.example" not in r.text and "[REDACTED]" in r.text
    assert r.headers["x-guardrail-redactions"] == "1"
    assert "bob@corp.example" not in s.text and "[REDACTED]" in s.text


def test_sqlite_lock_is_fast_503_not_500(providers, tmp_path):
    import sqlite3

    app = _composed(providers, tmp_path, sqlite_busy_timeout_ms=200)
    with run_uvicorn(app) as url:
        blocker = sqlite3.connect(str(tmp_path / "c.db"), isolation_level=None)
        blocker.execute("BEGIN EXCLUSIVE")  # another process holds the write lock
        try:
            t0 = time.perf_counter()
            r = chat(url)
            elapsed = time.perf_counter() - t0
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()
    assert_error_envelope(r, 503, "service_unavailable", "rate_limiter_unavailable")
    assert elapsed < 2


def test_stream_heartbeat_during_idle(providers, tmp_path):
    providers["p"].update(mode="stream_slow_first", delay_s=1.2)
    with run_uvicorn(_composed(providers, tmp_path, heartbeat_s=0.3)) as url:
        # TTFT deadline is 3 s, so the primary commits after 1.2 s; heartbeats only start after commit
        r = chat(url, stream=True)
    providers["p"].update(mode="ok", delay_s=0.0)
    content = "".join(
        c["delta"].get("content", "") for i in _sse_items(r.text) if isinstance(i, dict) for c in i.get("choices", [])
    )
    assert r.status_code == 200 and content.strip() == "Hello from primary"
