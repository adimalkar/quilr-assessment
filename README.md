# AI Field Developer / Solutions Engineer: Assessment Submission

Four production-minded components for securing and operating AI agents and LLM traffic,
written in Python 3.13. Each task has its own folder, and each one is verified by
black-box tests that run against **real processes and sockets**, not just mocked functions.

| # | Component | Folder | Tests | One-line pitch |
|---|-----------|--------|------:|----------------|
| 1 | MCP server (stdio) | [`task1_mcp_server/`](task1_mcp_server) | 59 | stdout is reserved for JSON-RPC **at the file-descriptor level**. Strict Pydantic schemas are the single source of truth. |
| 2 | MCP security gateway | [`task2_mcp_gateway/`](task2_mcp_gateway) | 111 | JWT role auth plus `admin_*` tool policy that resists look-alike names, duplicate/case-variant keys, batch smuggling, header mismatch and session hijacking. |
| 3 | Streaming PII guardrail | [`task3_llm_guardrail/`](task3_llm_guardrail) | 364 | Exact incremental redaction. Holds back the **provably minimal** suffix, so TTFT stays about one token and memory stays O(1). |
| 4 | Rate limit + fallback router | [`task4_model_router/`](task4_model_router) | 44 | Reserve-then-reconcile token limiter on SQLite, a 3000 ms **total** deadline, failover, and a cooldown breaker. |

```bash
uv sync                      # Python 3.13, deps pinned in uv.lock
uv run pytest                # 578 tests, ~50 s
```

---

## How this was built (3 iterations)

1. **Build**: implemented all four tasks to the brief, with tests.
2. **Prior-art sweep**: surveyed MCP gateways (IBM ContextForge, agentgateway, Envoy AI Gateway, Docker MCP
   Gateway, Lasso), LLM gateways (LiteLLM, Portkey, Kong AI Gateway, Bifrost, Envoy), and guardrails (Presidio,
   NeMo Guardrails, Guardrails AI, LLM Guard). Also read the MCP spec revisions 2025-06-18, 2025-11-25 and 2026-07-28.
   Every gap found was folded back in. The table at the end shows where this design deliberately differs from those projects.
3. **Adversarial review**: two independent reviewers attacked the code with concrete, executed exploits.
   They found **19 confirmed defects**. None let a viewer run an `admin_` tool, and none broke the redactor's
   chunking invariance (200k fuzzed splits). All 19 are fixed, and each has a regression test. See
   [Iteration 3: what the adversarial review found](#iteration-3-what-the-adversarial-review-found).

---

## Task 1: MCP server with strict validation and stdio isolation

```bash
uv run python -m task1_mcp_server.server                       # speaks MCP over stdio
npx @modelcontextprotocol/inspector uv run python -m task1_mcp_server.server   # optional: interactive UI
```

**Tools**
- `get_customer_record(customer_id)`: read-only.
- `trigger_refund(customer_id, amount, reason)`: destructive, not idempotent, and annotated that way so clients ask for confirmation.

Both tools advertise an `inputSchema` and an `outputSchema`. Each schema is generated from the Pydantic model that validates the call.

### STDIO isolation (enforced, not merely "remembered")
`_claim_stdio()` in [`server.py`](task1_mcp_server/server.py) works at the file-descriptor level:
1. It duplicates the real stdout onto a private, non-inheritable fd that **only the transport writer** holds.
2. It then calls `dup2(stderr, 1)`, so fd 1 becomes stderr.

As a result, `print()`, `sys.stdout.write`, `os.write(1, …)`, C extensions and **child processes** can no longer reach the wire.
The test `test_stray_prints_never_reach_stdout` injects all five kinds of rogue output into a live tool handler. It then
asserts that every stdout byte is a JSON-RPC frame and that every rogue line appeared on stderr.

### Validation & error mapping
Models use `strict=True`, `extra="forbid"` and `allow_inf_nan=False`.

**Field rules**
- `customer_id`: matches `^CUST-[0-9]{5}$`. The pattern uses Rust-regex semantics, so `"CUST-12345\n"` and Arabic-Indic digits are rejected.
- `amount`: must be `> 0`, at most `1e6`, and have at most 2 decimals. The decimal check uses `Decimal(str(x))`, not float maths.
- `reason`: whitespace is stripped *before* the 10–500 length check.

**Error mapping**

| Situation | Response |
|---|---|
| not JSON / invalid UTF-8 | `-32700 Parse error` (`id: null`) |
| JSON but not JSON-RPC, a batch array, an invalid id (`true`, `1.0`, `null`…), an unpaired surrogate, a >4000-digit number, or a frame > 4 MB | `-32600 Invalid Request` (id echoed when recoverable). Oversized frames are discarded in bounded chunks and never buffered whole. |
| unknown method | `-32601` |
| unknown tool, schema-invalid arguments | `-32602` with `data.errors[] = {field, type, message}`. **Input values are never echoed back.** |
| business-rule failure (unknown customer, refund > balance) | normal result with `isError: true` (a tool *execution* error, which the model can act on) |
| unexpected exception | `-32603`, generic message; traceback goes to stderr only |

The SDK silently drops undecodable lines, so [`transport.py`](task1_mcp_server/transport.py) adds the framing errors
in the first two rows. It never replies to notifications, even malformed ones, and handles EPIPE cleanly.

> **Spec note.** MCP 2025-11-25 (SEP-1303) recommends returning *argument-validation* failures as `isError` tool
> results so models can self-correct. This brief requires JSON-RPC errors, so `-32602` is the default.
> `MCP_VALIDATION_ERRORS=tool` switches to the SEP-1303 behaviour, and both modes are tested.

Both protocol eras are tested end to end: the `initialize` handshake (2025-11-25) and the stateless
`server/discover` envelope (2026-07-28). Interop is also tested with the official SDK `ClientSession`.

---

## Task 2: MCP security gateway (auth + tool-level authorization)

```bash
uv run uvicorn --factory task2_mcp_gateway.mock_downstream:app_factory --port 9001   # real SDK MCP server
GATEWAY_JWT_SECRET=$(openssl rand -hex 32) GATEWAY_DOWNSTREAM_URL=http://127.0.0.1:9001/mcp \
  uv run uvicorn --factory task2_mcp_gateway.gateway:app_factory --port 9000
```

```
agent ── POST /mcp ──► authenticate ─► strict parse ─► policy ─► forward (stream JSON/SSE as-is) ─► downstream MCP
          Bearer JWT       │               │              │
                           401          400 -32700/   tools/call admin_* && role≠admin
                                        -32600/-32020   → 200 {"error":{"code":-32001,"message":"Unauthorized Tool Call"}}
                                                          downstream is never contacted
```

- **Auth**: HS256 JWT. `alg` is pinned, which rejects `alg:none` and algorithm-confusion tokens. `exp`, `sub` and `role` are required;
  `iss`/`aud` are checked when configured; the secret must be at least 32 bytes. `role ∈ {admin, viewer}`, so any other role is denied.
  Repeated `Authorization` headers are rejected.
- **`tools/list`** is forwarded transparently, as the brief asks. `GATEWAY_FILTER_TOOLS_LIST=1` optionally hides `admin_*`
  tools from viewers (least-privilege discovery, as agentgateway and Envoy do). Enforcement on `tools/call` applies either way.
- **The `-32001` response** is sent with HTTP 200, so MCP SDK clients surface the JSON-RPC code instead of a transport error.
  It carries `data: {reason: "forbidden_tool", tool, required_role}`, which tells it apart from other uses of -32001 in the ecosystem.

### Bypass attempts that are blocked (each has a test)

| Attack | Defence |
|---|---|
| `ADMIN_x`, `ａｄｍｉｎ_x` (full-width), `admin​_x` (zero-width) | NFKC + strip `Cf` chars + casefold before the prefix check; then enforce the MCP tool-name charset `[A-Za-z0-9_.-]{1,128}` |
| `{"name":"get_x","name":"admin_y"}` | duplicate JSON keys are rejected; parsers disagree on first vs last |
| `{"name":"get_x","Name":"admin_y"}` / `"Method"` | case-variant protocol keys are rejected; case-insensitive decoders bind them (Envoy AI Gateway smuggling, Go SDK CVE-2026-27896) |
| what was authorised ≠ what is executed | the gateway forwards a **re-serialisation of the exact message it authorised**, never the raw bytes |
| batch `[allowed, admin_call]` | batches are rejected with `-32600` and nothing is forwarded (MCP removed batching in 2025-06-18; SDK servers reject arrays anyway) |
| `tools/call` as a notification (no `id`) | rejected and never forwarded; there would be no response to deny |
| `Mcp-Method` / `Mcp-Name` header ≠ body (2026-07-28) | `400 / -32020 HeaderMismatch`; handles the `=?base64?…?=` encoding |
| reusing another user's `Mcp-Session-Id` | clients receive `<downstream-id>.<HMAC(sub, id)>`. Anyone else's, unsigned or forged ids get 404. The binding is stateless, so it survives restarts and works across replicas ([`sessions.py`](task2_mcp_gateway/sessions.py)) |
| repeating a header (`Mcp-Session-Id: decoy` + `Mcp-Session-Id: <victim>`) | any repeated `Mcp-*` / forwarded / `Authorization` header gets 400. The value that is checked is the value that is forwarded |
| unpaired surrogates, 5000-digit numbers | `-32600`, not a 500; forwarded JSON is pure ASCII |
| exhausting the downstream pool with GET streams | long-lived streams use a separate pool and are capped per subject (429) |
| browser DNS-rebinding | `Origin` must be allow-listed when present (`GATEWAY_ALLOWED_ORIGINS`) |
| token passthrough | the client's `Authorization`, cookies and hop-by-hop headers are stripped; an optional separate downstream credential is injected |

**Proxying**
- Responses stream byte-for-byte. Both JSON and SSE work (tests run against both downstream modes).
- `Mcp-Session-Id` is relayed in both directions (signed), and GET (SSE stream) and DELETE (session end) are proxied.
- Headers are forwarded per RFC 9110: hop-by-hop headers and the headers named in `Connection` are dropped, while `Content-Encoding` and repeated `Set-Cookie` headers are preserved.
- The gateway issues its own request ids, so a client-supplied `X-Request-Id` cannot spoof audit-log correlation.
- SSE responses get `X-Accel-Buffering: no`. On client disconnect, the downstream connection is closed in `finally`.
- Downstream failures return a sanitised `502 / -32002` or `504 / -32003`. Every decision is written as a JSON audit line (`sub, role, method, tool, decision, reason, rid`), and the token is never logged.

The end-to-end test drives the **official SDK Streamable HTTP client** through the gateway to a **real SDK MCP server**.

---

## Task 3: LLM gateway streaming guardrail (PII redaction)

```bash
uv run uvicorn --factory task3_llm_guardrail.mock_llm:app_factory --port 8301        # OpenAI-compatible mock
uv run uvicorn --factory task3_llm_guardrail.gateway:app_factory --port 8300
# or point it at a real provider:
GUARD_UPSTREAM_BASE_URL=https://api.openai.com/v1 GUARD_UPSTREAM_API_KEY=sk-... uv run uvicorn --factory ...
```

### The algorithm ([`redactor.py`](task3_llm_guardrail/redactor.py))
Deltas arrive as fragments like `"Contact jo"`, `"hn@exam"`, `"ple.com"`. After each delta, the redactor computes the
**hold point**: the smallest `i` such that `buf[i:]` could still be the beginning of a PII match. It does this with the
`regex` module's `fullmatch(buf, pos=i, partial=True)`. Everything before `i` is provably final:

- No match can start there and still grow.
- Every match found there already has its final extent.

So those matches are redacted and the text is released. Only `buf[i:]` is kept, which in practice is the last
word or number. This gives three properties:

- **Chunking invariance.** For *any* split of a text into deltas, the streamed output equals `redact_text(full_text)`.
  This is property-tested over 300 random texts × random chunkings, plus char-by-char streaming. It means streaming
  never leaks anything that batch mode would catch.
- **Bounded memory.** Every quantifier is bounded, so the buffer is at most `MAX_MATCH_LEN` (640 chars) on any input.
  A >100 KB stream plus a 10 KB unbroken token stays within that bound (tested).
- **Low latency.** Measured at **~8.5 µs per delta** (1.1 MB in 4-char deltas), with a peak buffer of 24 chars on realistic text.
  Throughput stays linear on pathological 200 KB runs of digits or letters. The gateway test asserts that the first redacted delta
  reaches the client in under 250 ms of a 1 s stream.

**Patterns**
- Email: RFC 5321 length limits; Unicode-aware, so IDN domains like `exämple.com` are caught.
- US SSN: Presidio's invalid ranges are excluded (area 000/666, group 00, serial 0000), and separators (`-`, space, `.`, en dash) must be consistent.
  Unseparated 9-digit numbers are flagged only when labelled ("SSN is …", "social security number: …"), since bare 9-digit numbers are mostly order ids.
- Payment cards: 13–19 digits, or 4-4-4-x / Amex 4-6-5 groups with any separators, confirmed by a **Luhn** check (Unicode digits are handled via `unicodedata`).
  Order numbers and timestamps pass through.

### Stream handling ([`sse.py`](task3_llm_guardrail/sse.py), [`stream_guard.py`](task3_llm_guardrail/stream_guard.py), [`gateway.py`](task3_llm_guardrail/gateway.py))

**Decoding the stream**
- The SSE decoder follows WHATWG rules, modelled on openai-python's decoder.
- It handles CR, LF and CRLF, including a CRLF split across two network reads, and multi-byte UTF-8 split across reads. Every split point is tested.

**Redaction coverage**
- **Every string in every choice is redacted**, whatever its key: `content` (including list-valued parts), `refusal`,
  `reasoning(_content)`, `tool_calls[].function.arguments`, legacy `function_call`, `audio.transcript`, and any field a
  provider adds later. An allowlist of "known text fields" is how PII leaks. Short identifiers (`role`, `id`, `name`, …)
  are redacted in one shot, never held back, so tool names are never split across chunks.
- There is one redactor per `(choice, JSON path)`. List items are keyed by `index`, else `id`, else position, so `n>1`
  choices, parallel tool calls and multi-part content never share a buffer.
- **`logprobs` are stripped**, because they carry raw token text that would bypass redaction.

**Re-framing**
- Released text goes out in the envelope of the chunk that completed it.
- Chunks that are entirely held back are suppressed.
- Held text is flushed into the `finish_reason` chunk, or into a synthetic chunk before `[DONE]`.

**Failure handling**
- **Fail closed.** Any guard error ends the stream with an in-stream `guardrail_error` event. Raw text is never released as a fallback.
- Upstream errors (non-200 responses, in-stream `{"error"}`, truncation, stalls) become sanitised OpenAI-style errors. Provider internals are never relayed.
- **Client disconnect closes the upstream connection**, so the provider stops generating and billing. This is asserted against the mock provider's cancellation counter.

---

## Task 4: token-aware rate limiter + model fallback router (SQLite)

```bash
uv run uvicorn --factory task4_model_router.mock_providers:app_factory --port 8401   # primary
uv run uvicorn --factory task4_model_router.mock_providers:app_factory --port 8402   # secondary
ROUTER_DB_PATH=./data/router.db ROUTER_TENANTS="tenant-key-1:acme,tenant-key-2:globex:100000" \
  uv run uvicorn --factory task4_model_router.app:app_factory --port 8400
```

### Limiter ([`limiter.py`](task4_model_router/limiter.py)): sliding log of weighted entries
- **Data model.** One SQLite row per admitted request: `(key_hash, ts_ms, tokens)`. Usage is the sum over `ts_ms > now − 60 s`.
  This is exact, with no fixed-window boundary burst (tested), and costs one row per request rather than one per token.
- **Reserve → reconcile → refund.**
  - Before the call, the limiter reserves `prompt_estimate + max_tokens`, which is OpenAI's own accounting rule.
    Concurrent requests therefore cannot jointly overshoot the limit. Kong and Envoy charge *after* the response, so they can.
  - After the response, the row is set to the provider's `usage.total_tokens`.
  - On total failure the reservation is refunded. A client that disconnects mid-stream pays for what was generated.
- **Atomicity.** Check-and-insert runs in `BEGIN IMMEDIATE`, with WAL and `busy_timeout`. Tests: 200 threads, and **4 OS processes
  racing on one DB file**, admit exactly the budget and never overshoot. All SQLite calls run via `asyncio.to_thread`, so the event loop never blocks.
- **Eviction.** Expired rows for a key are deleted inside the same transaction that reads them. A periodic sweep handles idle keys.
- **Retry-After.** The limiter walks entries oldest-first until enough tokens would be freed for *this* request, then
  computes when that entry expires. The test checks the exact millisecond.
- **Non-retryable requests.** A request whose reservation exceeds the tenant's whole limit gets **400 `request_exceeds_token_limit`**, not
  a 429 that clients would retry forever.
- **Stored keys.** API keys are stored only as SHA-256 hashes (tested by scanning the raw DB and WAL bytes). Per-tenant limit overrides are supported.

### Router ([`router.py`](task4_model_router/router.py))

**Deadlines**
- **The 3000 ms is a total deadline** (`asyncio.timeout`) covering connect, headers and body. httpx's `read` timeout measures
  the gap between reads, so a provider that trickles one byte every 0.5 s would never trigger it. A test reproduces exactly that case.
- When the deadline fires, the request task is cancelled and httpx closes that connection. A late primary response can therefore never
  be observed, so there is no double answer or double charge.
- **Streaming uses a time-to-first-token deadline.** The router does not commit to a provider until it sends its first *content*
  delta (a role-only delta doesn't count). After commit, failover is impossible, so a later failure becomes a sanitised in-stream error.

**Failover policy**
- Fail over on: 429, 408, 5xx, 529, transport errors and timeouts, and 401/402/403/404 (that provider's credentials, billing or model config).
- Don't fail over on: 400, 413, 422. The backup would reject the same bad request.

**Cooldown circuit breaker**
- A 429 parks the provider for its `Retry-After`, capped at 60 s. The router never sleeps inside a request.
- Timeouts and 5xx park it after 3 consecutive failures. While parked, traffic goes straight to the backup instead of
  paying 3 s per request (tested by counting primary calls).
- When the cooldown expires, the breaker goes **half-open**: exactly one probe request is let through. Success closes it;
  failure re-opens it at once. That way a recovering provider isn't hit by a burst of requests that all pay the timeout.
- Failures *before the first token* fail over, whatever they are: an in-stream `{"error"}`, invalid bytes, or a decode error.

**Accounting edge cases** (all from the adversarial review)
- The reservation covers `n` choices: `prompt + n × max_tokens`.
- The tenant is authenticated *before* the body is read. The body size is capped (413), so unauthenticated callers can't make the gateway buffer large bodies or learn its schema.
- Every failure path refunds the reservation, including unexpected exceptions and cancellation.
- If a stream outlives the 60 s window and its row is evicted, the usage is re-inserted, never forgiven.
- Charged output includes tool-call arguments and reasoning, not just `content`.

### Errors ([`errors.py`](task4_model_router/errors.py))
Every error is an OpenAI-compatible envelope, `{"error":{"message","type","code","param","request_id"}}`. This includes
validation errors, malformed JSON, 404/405 and unhandled exceptions. The messages are written by the gateway, never copied from
upstream. Tests feed the mock providers a body containing a fake traceback, org id and API key, and assert that none of it reaches the client.

**Response headers**
- Every response carries `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-tokens` and `x-ratelimit-reset-tokens`, plus `x-request-id`.
- Successful responses also carry `x-gateway-provider` and `x-gateway-fallback`.
- 429 responses also carry `Retry-After`.

---

## Where this differs from popular prior art

| Concern | Common approach | This submission |
|---|---|---|
| Streaming PII | LiteLLM/Presidio buffers the **whole** stream, then masks; on error it **yields raw chunks** | exact incremental holdback; fails closed |
| Stream windows | NeMo `chunk_size`/`context_size` (and `stream_first` emits *before* checking); Guardrails AI waits for sentences | minimal provable holdback (≈1 token); checked before emit |
| Fields redacted | `content` only | **every string** in every choice (tool calls, legacy `function_call`, list content, audio transcripts, unknown future fields); logprobs stripped |
| Token limits | Kong / Envoy AI Gateway charge after the response, so admitted requests overshoot | reserve up front, reconcile to actual |
| Timeouts | per-read httpx/requests timeouts | total deadline; TTFT deadline for streams |
| MCP authz | prefix/glob match on the raw name | normalised name, parser-differential defences, header/body consistency, session binding |

## Iteration 3: what the adversarial review found

Two reviewers, given only the code and the brief, wrote and ran exploit scripts. Every confirmed finding is fixed and has a regression test:

| Sev | Component | Finding | Fix |
|---|---|---|---|
| High | MCP gateway | `Mcp-Session-Id: decoy` + `Mcp-Session-Id: <victim>`: the first header was checked, the last forwarded, so bob rode alice's session | repeated headers rejected; HMAC-signed session ids |
| High | MCP server | `"id":"\ud800"` (lone surrogate) crashed the server | rejected with `-32600`; serialisation guarded |
| High | MCP server | the 4 MB line cap was checked only *after* buffering the whole line (300 MB → +312 MB RSS) | bounded `readline`, overflow drained in chunks |
| High | PII guardrail | PII leaked via legacy `function_call`, list-valued `content`, `audio.transcript` | generic every-string walker |
| High | Router | invalid bytes before the first token → 500, no failover, reservation stuck | any pre-commit failure fails over; catch-all refund |
| High | Router | `n=10` reserved only 1× `max_tokens`, so a window reached 80k/50k | reserve `n × max_tokens` |
| Med | MCP server | `"id": true / 1.0 / {} / null` silently dropped (client hangs) | `-32600` |
| Med | MCP gateway | session binding was in-memory: lost on restart, and unknown ids were allowed | stateless HMAC binding, fail-closed |
| Med | MCP gateway | 200 GET streams by one viewer exhausted the pool, so admin POSTs timed out | separate stream pool and per-subject cap |
| Med | Router | the body was parsed before auth, with no size cap (150 MB unauthenticated) | auth first; 413 cap |
| Med | Router | an in-stream error before the first token was "committed" instead of failing over | fails over |
| Med | PII guardrail | parallel tool calls without `index` shared one buffer | keyed by `index` / `id` / position |
| Med | PII guardrail | `SSN-123-45-6789`, `4111.1111…`, en dashes, mixed separators, IDN emails, full-width digits missed | patterns widened; Unicode Luhn |
| Low | various | reason of 10 zero-width spaces accepted; gzip `Content-Encoding` stripped; batches skipped header checks; 5000-digit ints → wrong code; bad log level crashed; wrong `fallback` flag; Set-Cookie merged; spoofable request id; no half-open probe; evicted rows forgiven | all fixed |

While widening the SSN patterns I found a quirk in the `regex` module: in partial-matching mode, an alternation that
contains a variable-length lookbehind reports spurious partial matches. That made the holdback grow to 700+ chars.
The labelled-SSN rule now captures its label instead of looking behind for it, and the hold point is computed per pattern.
The bounded-memory test caught the problem.

## Known limitations / next steps
- **Task 3 detection is regex-based.** That's deterministic and fast, but names and addresses would need an NER model, for example
  Presidio run per *released* segment. The holdback design makes that a drop-in addition. Input-side (prompt) redaction is not enabled.
- **Task 4 token estimates are a heuristic** (chars/4 plus per-message overhead). Reconciliation corrects them; `tiktoken`
  could be dropped in for exact pre-counts. Circuit-breaker state is per process, while the limiter is shared across processes via SQLite.
  At very high QPS, the per-request rows could be swapped for a sliding-window counter or GCRA, accepting approximation for O(1) state.
- **Task 2 does not authorize MCP `tasks/*` (2025-11-25) by subject.** Tool calls augmented with tasks are still policy-checked
  as `tools/call`, but fetching a task result by id relies on the downstream scoping it to the session.
- **Task 2 JWTs are HS256.** A production deployment would use the IdP's JWKS (RS256/ES256) and OAuth 2.1 resource-server metadata, per the MCP authorization spec.
