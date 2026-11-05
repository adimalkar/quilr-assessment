# AI Field Developer / Solutions Engineer: Assessment Submission

[![ci](https://github.com/adimalkar/quilr-assessment/actions/workflows/ci.yml/badge.svg)](https://github.com/adimalkar/quilr-assessment/actions/workflows/ci.yml)

Production-minded components for securing and operating AI agents and LLM traffic, written in Python 3.12+.
Every claim here is backed by a black-box test that runs against **real processes and sockets**. The test names are in the
**[rubric map](docs/RUBRIC.md)**, which quotes each evaluation criterion from the brief and points to the code and tests that cover it.

| # | Component | Folder | Tests | One-line pitch |
|---|-----------|--------|------:|----------------|
| 1 | MCP server (stdio) | [`task1_mcp_server/`](task1_mcp_server) | 66 | stdout is reserved for JSON-RPC **at the file-descriptor level, before any third-party import**, and a negative-control test proves the check can fail. |
| 2 | MCP security gateway | [`task2_mcp_gateway/`](task2_mcp_gateway) | 115 | JWT roles, `admin_*` policy that resists look-alike names / key smuggling / header mismatch / session hijacking, plus tool-result DLP and rug-pull pinning. |
| 3 | Streaming PII guardrail | [`task3_llm_guardrail/`](task3_llm_guardrail) | 624 | Exact incremental redaction of **every** text field. It holds back the provably minimal suffix, so TTFT grows by about one word and memory stays bounded. |
| 4 | Rate limit + fallback router | [`task4_model_router/`](task4_model_router) | 48 | Reserve-then-reconcile token limiter on SQLite (atomic across processes), a 3000 ms **total** deadline, a half-open breaker, and the Task 3 guard built in. |
| 5 | Zero-trust debugging scenario | [`task5_zero_trust/`](task5_zero_trust) | 2 | A step-by-step playbook. Its main hypothesis (idle-timeout middleboxes) is **reproduced and fixed in a test**. |
| | End-to-end composition | [`tests/test_e2e_composition.py`](tests/test_e2e_composition.py) | 2 | SDK client → Task 2 gateway → Task 1 server over Streamable HTTP. |

## Quick start

```bash
uv sync                       # deps pinned in uv.lock (Python 3.12 or 3.13)
make test                     # 857 tests, ~60 s   (= uv run pytest)
make demo                     # narrated walkthrough of all tasks against live servers → docs/DEMO.md
make serve-task2              # or task3 / task4 / all: start a stack on fixed ports, print tokens + curls
make bench                    # reproduce the numbers below → docs/BENCHMARKS.md
make inspector                # Task 1 in the MCP Inspector UI
GATEWAY_JWT_SECRET=$(openssl rand -hex 32) docker compose up --build   # everything in containers
```

To use the Task 1 server from an MCP client (Claude Desktop, Claude Code, Cursor, …):
```json
{"mcpServers": {"customer-support": {"command": "uv", "args": ["--directory", "/path/to/repo", "run", "python", "-m", "task1_mcp_server"]}}}
```
or `claude mcp add customer-support -- uv --directory /path/to/repo run python -m task1_mcp_server`.

**[docs/DEMO.md](docs/DEMO.md)** shows captured output for every scenario: allowed and denied calls, smuggling attempts, a redacted
stream with per-delta timings, and failover with headers. If you don't want to run anything, start there.

---

## How this was built (4 iterations)

1. **Build**: all tasks implemented to the brief, with black-box tests.
2. **Prior-art sweep**: I surveyed and borrowed ideas from:
   - MCP gateways: IBM ContextForge, agentgateway, Envoy AI Gateway, Docker MCP Gateway, Lasso;
   - LLM gateways: LiteLLM, Portkey, Kong AI Gateway, Bifrost;
   - guardrails: Presidio, NeMo Guardrails, Guardrails AI, LLM Guard;
   - the MCP spec revisions 2025-06-18, 2025-11-25 and 2026-07-28.
3. **Adversarial review**: independent reviewers attacked the code with executed exploits and found **19 defects**.
   [All are fixed, each with a regression test](#iteration-3-what-the-adversarial-review-found).
4. **Peer benchmark**: I read the 16 most substantial public submissions of this assessment, statically and without running them.
   That surfaced two more redaction edge cases, which are now fixed and covered by an independent ground-truth oracle test.
   I then took the best ideas for reviewer experience (demo, rubric map, runnable stacks, benchmarks, CI), plus Task 5,
   which appears in some copies of the brief.

---

## Task 1: MCP server with strict validation and stdio isolation

**Entry point:** `uv run python -m task1_mcp_server` (stdio), or `task1_mcp_server.http_app:app_factory` (Streamable HTTP, used behind Task 2).

**Tools**
- `get_customer_record(customer_id)`: read-only.
- `trigger_refund(customer_id, amount, reason, idempotency_key?)`: destructive and annotated as such.
  The optional **idempotency key** makes an agent's retry after a timeout return the original refund instead of paying twice.

Both tools advertise an `inputSchema` and an `outputSchema`, generated from the Pydantic models that validate every call.

### STDIO isolation (enforced, not merely "remembered")
[`stdio_guard.py`](task1_mcp_server/stdio_guard.py) imports only the standard library. It duplicates the real stdout onto a
private, non-inheritable fd held by the transport writer, then runs `dup2(stderr, 1)`. As a result, `print()`,
`sys.stdout.write`, `os.write(1, …)`, C extensions and **child processes** all land on stderr. [`__main__.py`](task1_mcp_server/__main__.py)
claims the wire *before* importing the MCP SDK and pydantic, so even a dependency's import-time `print()` can't corrupt the stream.

**Tests:**
- Rogue output of every kind is injected into a live tool handler, and an import-time print into a dependency. Every stdout byte must still be JSON-RPC.
- A **negative control** shows that the same import-time print *does* corrupt the wire if the wire is claimed too late, so the check can fail.
- ruff's `T20` rule bans `print` in library code.

### Validation & error mapping
Models use `strict=True`, `extra="forbid"` and `allow_inf_nan=False`.

**Field rules**
- `customer_id`: `^CUST-[0-9]{5}$`. Rust-regex semantics, so a trailing `\n` and Arabic-Indic digits are rejected.
- `amount`: `> 0`, at most `1e6`, at most 2 decimals. The decimal check uses `Decimal(str(x))`, not float maths.
- `reason`: 10–500 characters, counted after stripping whitespace. Zero-width and control characters are rejected.

| Situation | Response |
|---|---|
| not JSON / invalid UTF-8 | `-32700 Parse error` (`id: null`) |
| not JSON-RPC: batch array, bad id (`true`, `1.0`, `null`…), unpaired surrogate, a >4000-digit number, or a frame > 4 MB | `-32600 Invalid Request`. Oversized frames are discarded in bounded chunks and never buffered. |
| unknown method | `-32601` |
| unknown tool, schema-invalid arguments | `-32602` with `data.errors[] = {field, type, message}`. **Input values are never echoed back.** |
| business-rule failure (unknown customer, refund > balance, key reused with different arguments) | result with `isError: true` (a tool *execution* error, which the model can act on) |
| unexpected exception | `-32603`, generic message; traceback goes to stderr only |

> **Spec note.** MCP 2025-11-25 (SEP-1303) recommends returning argument-validation failures as `isError` results so models
> can self-correct. The brief asks for JSON-RPC errors, so `-32602` is the default, and `MCP_VALIDATION_ERRORS=tool` switches to SEP-1303.
> Both protocol eras are tested: the `initialize` handshake (2025-11-25) and the stateless `server/discover` envelope (2026-07-28).

---

## Task 2: MCP security gateway (auth + tool-level authorization)

```
agent ── POST /mcp ──► authenticate ─► strict parse ─► policy ─► forward (stream JSON/SSE as-is) ─► downstream MCP
          Bearer JWT       │               │              │
                           401          400 -32700/   tools/call admin_* && role≠admin
                                        -32600/-32020   → 200 {"error":{"code":-32001,"message":"Unauthorized Tool Call"}}
                                                          downstream is never contacted
```

**Running it:** `make serve-task2` starts Task 1 behind the gateway, mints tokens and prints curl commands.
For other tokens: `uv run python -m task2_mcp_gateway.devtoken --role viewer --sub bob`.

**Auth**
- HS256 JWT with a pinned `alg` (rejects `alg:none` and algorithm confusion).
- `exp`, `sub` and `role` are required; `iss`/`aud` are checked when configured; the secret must be at least 32 bytes.
- `role ∈ {admin, viewer}`; anything else is denied.

**`tools/list`** is forwarded transparently, as the brief asks. `GATEWAY_FILTER_TOOLS_LIST=1` optionally hides `admin_*` tools from viewers.

**The -32001 response** is sent with HTTP 200, so SDK clients surface the JSON-RPC code. Its
`data: {reason, tool, required_role}` tells it apart from other uses of -32001 in the ecosystem.

### Bypass attempts that are blocked (each has a test)

| Attack | Defence |
|---|---|
| `ADMIN_x`, `ａｄｍｉｎ_x` (full-width), `admin​_x` (zero-width) | NFKC + strip `Cf` chars + casefold before the prefix check; then the MCP tool-name charset `[A-Za-z0-9_.-]{1,128}` |
| `{"name":"get_x","name":"admin_y"}` | duplicate JSON keys rejected (parsers disagree on first vs last) |
| `{"name":"get_x","Name":"admin_y"}` / `"Method"` | case-variant protocol keys rejected (case-insensitive decoders bind them; cf. Go SDK CVE-2026-27896) |
| authorised ≠ executed | the gateway forwards a **re-serialisation of exactly what it authorised**, never the raw bytes |
| batch `[allowed, admin_call]` | batches → `-32600`, nothing forwarded (MCP removed batching in 2025-06-18) |
| `tools/call` as a notification | rejected, never forwarded |
| `Mcp-Method` / `Mcp-Name` header ≠ body (2026-07-28) | `400 / -32020 HeaderMismatch`, including `=?base64?…?=` values |
| another user's `Mcp-Session-Id` | ids are issued as `<downstream-id>.<HMAC(sub, id)>`; foreign, unsigned or forged ids → 404. The binding is stateless, so it survives restarts and works across replicas |
| `Mcp-Session-Id: decoy` + `Mcp-Session-Id: <victim>` | any repeated security-relevant header → 400; the value checked is the value forwarded |
| unpaired surrogates, 5000-digit numbers | `-32600`, not a 500; forwarded JSON is pure ASCII |
| pool exhaustion through GET streams | separate stream pool, per-subject cap (429) |
| DNS rebinding / token passthrough | `Origin` allow-list; the client's `Authorization` and cookies are never forwarded (an optional separate downstream credential is injected) |

**Beyond the brief**, both opt-in and both a natural fit for a data-security product:
- **Tool-result DLP** (`GATEWAY_REDACT_TOOL_RESULTS=viewers|all`). Tool output passes through Task 3's redactor, so a viewer's agent
  sees `"email": "[REDACTED]"` while admins see the record in clear.
- **Tool-definition pinning** (`GATEWAY_PIN_TOOLS=1`). Each tool's description and schema are hashed when first seen. If the downstream
  later swaps in a different definition, such as a prompt-injection payload (a "rug pull"), that tool is removed from `tools/list`
  and calls to it are denied with `-32001 tool_definition_changed`.

**Proxying**
- Responses stream byte-for-byte (JSON and SSE; every test runs against both downstream modes).
- Header handling follows RFC 9110 (hop-by-hop and `Connection`-named headers dropped; `Content-Encoding` and multiple `Set-Cookie` kept).
- GET and DELETE are proxied, and SSE responses get `X-Accel-Buffering: no`.
- On client disconnect, the downstream connection is closed in `finally`.
- Errors are sanitised as `502 / -32002` and `504 / -32003`. The gateway generates its own request ids and writes one JSON audit line per decision; the token is never logged.

---

## Task 3: LLM gateway streaming guardrail (PII redaction)

**Running it:** `make serve-task3`. To use a real provider, set `GUARD_UPSTREAM_BASE_URL=https://api.openai.com/v1 GUARD_UPSTREAM_API_KEY=sk-…`.

### The algorithm ([`redactor.py`](task3_llm_guardrail/redactor.py))
Deltas arrive as fragments: `"Contact jo"`, `"hn@exam"`, `"ple.com"`. After each delta, the redactor computes the **hold
point**: the smallest `i` such that `buf[i:]` could still begin a PII match. It finds it with the `regex` module's
`fullmatch(buf, pos=i, partial=True)`, checked per pattern. Everything before `i` is provably final, so it is redacted and released.
Only the tail is kept, which in practice is the last word or number. Batch mode and streaming share **one scan rule**, so they can't diverge.

- **Chunking invariance.** For *any* split into deltas, the streamed output equals `redact_text(full_text)`. This is property-tested
  over 300 random texts and chunkings, plus char-by-char streaming. Reviewers ran 200k more random splits without a mismatch.
- **Independent ground-truth oracle.** 250 randomised documents get known PII injected at known positions, glued to random
  neighbours (letters, digits, punctuation, emoji). The streamed output must contain none of it. This test doesn't trust our own
  regexes, and it found a real overlap bug, now fixed: a rejected card candidate used to swallow a real card next to it.
- **Bounded memory.** Every quantifier is bounded, so the buffer stays at or under 640 chars on any input, pathological ones included (tested).
- **Latency.** The guardrail costs about **10 µs** of CPU per delta. Measured end to end, it adds about one held-back word to
  time-to-first-token. The worst case (a 50-char email opening the stream) is reported too. See [Benchmarks](#benchmarks).

> **"Isn't `regex` partial matching unsound?"** The usual pitfall is `search(partial=True)`, where a partial match can hide a
> complete match that starts later. We never use that: at each position we ask `fullmatch(pos=i, partial=True)`, meaning "could the
> text from here still be a match?", and we redact with an ordinary full scan. The invariance and oracle tests check exactly this property.

**Patterns**
- **Email:** RFC 5321 length limits, Unicode-aware so IDN domains are caught. A TLD is letters only, so `a@b.com5` is still caught.
- **US SSN:** Presidio's invalid ranges are excluded (000/666 area, 00 group, 0000 serial), and the separators (`-`, space, `.`, en dash)
  must be consistent. The separated shape is caught even when glued to digits or letters. Unseparated 9-digit numbers count
  only with a label ("SSN is …").
- **Cards:** 13–19 digits, or grouped 4-4-4-x / Amex 4-6-5, every candidate **Luhn-checked**. Grouped shapes glued to digits are
  trimmed to the Luhn-valid prefix (`94111 1111 1111 1111` → `9[REDACTED]`).

### Stream handling ([`sse.py`](task3_llm_guardrail/sse.py), [`stream_guard.py`](task3_llm_guardrail/stream_guard.py), [`gateway.py`](task3_llm_guardrail/gateway.py))

**Decoding and coverage**
- The SSE decoder follows WHATWG rules (CR/LF/CRLF, including CRLF and multi-byte UTF-8 split across reads; every split point is tested).
- **Every string in every choice is redacted**, whatever its key:
  - `content`, including list-valued parts;
  - `refusal` and `reasoning(_content)`;
  - `tool_calls[].function.arguments` and legacy `function_call`;
  - `audio.transcript`, and any field a provider adds later.

  An allowlist of "known text fields" is how PII leaks. Short identifiers (`role`, `id`, `name`) are redacted in one shot,
  never held back. **`logprobs` are stripped**, because they carry raw token text.
- One redactor per `(choice, JSON path)`, with list items keyed by `index`, then `id`, then position. `n>1` choices and parallel tool calls never share a buffer.

**Resilience**
- **Fail closed.** A guard error ends the stream with a `guardrail_error` event; raw text is never released.
- Upstream errors are sanitised, including provider 401/403, which are the gateway's own credential problem and become 502.
- **Heartbeats.** `: ping` comments during idle gaps keep proxies from cutting quiet streams (see Task 5).
- **Client disconnect closes the upstream connection**, so the provider stops generating and billing. This holds with heartbeats too, because the cleanup is shielded from anyio's level-triggered cancellation (tested against the mock's cancellation counter).

---

## Task 4: token-aware rate limiter + model fallback router (SQLite)

**Running it:** `make serve-task4`. The mocks can be driven by hand: `POST :8401/_mock/mode {"mode":"rate_limited"}`, `GET :8401/_mock/stats`, or a per-request `x-mock-mode` header.

### Limiter ([`limiter.py`](task4_model_router/limiter.py)): sliding log of weighted entries
- **Data model.** One SQLite row per admitted request: `(key_hash, ts_ms, tokens)`. Usage is the sum over the last 60 s.
  This is exact, with no fixed-window boundary burst (tested), and costs one row per *request*, not per token.
- **Reserve → reconcile → refund.**
  - It reserves `prompt_estimate + n × max_tokens` before the call (OpenAI's own rule), so concurrent requests can't jointly overshoot.
    Kong and Envoy charge after the response, so theirs can.
  - The row is reconciled to `usage.total_tokens` afterwards.
  - It is refunded on any failure path. A client that disconnects mid-stream pays for what was generated.
  - Usage is re-inserted if a long stream outlived its row.
- **Atomicity.** Check-and-insert runs inside `BEGIN IMMEDIATE`, with WAL. **4 OS processes racing on one DB file** admit exactly the budget.
  The bounded `busy_timeout` turns a locked DB into a fast `503 rate_limiter_unavailable`, never a hang or a 500.
  SQLite calls run via `asyncio.to_thread`, so they never block the event loop.
- **Retry-After.** It walks entries oldest-first to the exact millisecond this request would fit (tested).
- **Non-retryable requests.** A request bigger than the whole limit gets `400 request_exceeds_token_limit`, not a 429 that clients would retry forever.
- **Auth and storage.**
  - API keys are stored only as SHA-256 hashes (tested on the raw DB bytes).
  - The tenant is authenticated before the body is read, and the body size is capped.

### Router ([`router.py`](task4_model_router/router.py))

**Deadlines**
- **The 3000 ms is a total deadline** (`asyncio.timeout`) over connect, headers and body. httpx's `read` timeout measures the gap
  between reads, so a provider trickling one byte every 0.5 s would never trip it. A test reproduces that case.
- Cancellation closes the connection, so a late primary answer can never be observed: no double answer, no double charge.
- **Streams use a time-to-first-token deadline.** The router commits to a provider only after its first content delta. Anything failing
  before that (in-stream errors, bad bytes) fails over; anything after it becomes a sanitised in-stream error.

**Failover policy**
- Fail over on 429, 408, 5xx, 529, transport errors and timeouts, and 401/402/403/404 (that provider's own config).
- Don't fail over on 400, 413 or 422: the backup would reject the same bad request.

**Cooldown circuit breaker**
- A 429 parks the provider for its `Retry-After` (capped at 60 s; the router never sleeps inside a request).
- 3 consecutive failures park it for 30 s.
- After the cooldown the breaker is **half-open**: exactly one probe gets through, so a recovering provider doesn't take a thundering herd.

**Composed with Task 3:** `ROUTER_REDACT_PII=1` runs the streaming guardrail on every routed response, so one endpoint
does rate limiting, failover and PII redaction (on by default in `docker-compose.yml`).

**Errors** ([`errors.py`](task4_model_router/errors.py))
- Every error is an OpenAI-compatible envelope, `{"error":{"message","type","code","param","request_id"}}`. That covers validation errors,
  bad JSON, 404/405, storage errors and unhandled exceptions.
- Tests feed the mocks a body containing a fake traceback, org id and API key, and assert that none of it reaches the client.
- Successful and 429 responses carry `x-ratelimit-{limit,remaining,reset}-tokens` (in OpenAI's format), successful ones also `x-gateway-provider` / `x-gateway-fallback`, and every response has `x-request-id`.

---

## Task 5: zero-trust debugging scenario

> "An enterprise client deploys our LLM Gateway and MCP Gateway inside a zero-trust corporate network behind a strict egress
> proxy. Agents experience intermittent connection drops during multi-turn tool execution, resulting in hanging requests."

Some copies of the brief carry only the heading ("troubleshooting zero-trust network deployments"), so it is answered here for
completeness. **[`task5_zero_trust/README.md`](task5_zero_trust/README.md)** covers:
- the first 15 minutes of a customer-visible incident;
- commands to run: curl phase timings through the proxy, `openssl s_client -proxy`, tcpdump/tshark RST analysis, `ss`/conntrack, `py-spy`;
- a table telling **mTLS vs SSE idle timeout vs context truncation** apart by signature;
- a transport and pooling redesign.

The remediations reference this code, not hypotheticals:
- heartbeats;
- total and TTFT deadlines;
- a separate stream pool;
- restart-proof HMAC sessions;
- idempotency keys for safe retries;
- async MCP tasks for long tools.

The main hypothesis is **reproduced in a test**. [`idle_proxy.py`](task5_zero_trust/idle_proxy.py) is a middlebox that drops
connections idle for more than 1 s. Through it, a stream whose model "thinks" for 2.5 s is dropped, and with heartbeats it completes.

---

## Benchmarks

`make bench` regenerates [docs/BENCHMARKS.md](docs/BENCHMARKS.md) (local loopback, single machine):

| | p50 | p95/p99 |
|---|---:|---:|
| Redactor CPU per 4-char delta (realistic text) | 9.9 µs | 34 µs (p99) |
| TTFT, provider direct | 1.4 ms | 2.3 ms |
| TTFT, through the guardrail (one word held back) | 13.4 ms | 15.1 ms |
| TTFT, through the guardrail, 50-char email *opens* the stream (worst case) | 197 ms | 199 ms |
| Failover with 50 concurrent requests to a provider that takes 10 s (budget 3000 ms) | 3112 ms | 3152 ms |

---

## Configuration

Every variable is listed in [`.env.example`](.env.example). The main ones:

| Variable | Default | Effect |
|---|---|---|
| `GATEWAY_JWT_SECRET` | (required) | HS256 secret, at least 32 bytes; also keys the session-binding HMAC |
| `GATEWAY_DOWNSTREAM_URL` | `http://127.0.0.1:9001/mcp` | downstream MCP server |
| `GATEWAY_REDACT_TOOL_RESULTS` | `off` | `viewers` / `all`: DLP on tool results |
| `GATEWAY_PIN_TOOLS` | `0` | quarantine tools whose definition changes |
| `GUARD_UPSTREAM_BASE_URL` | mock `:8301` | OpenAI-compatible provider |
| `GUARD_HEARTBEAT_S` / `ROUTER_HEARTBEAT_S` | `15` | SSE keep-alive interval (`0` disables) |
| `ROUTER_TPM` | `50000` | default tokens per minute per tenant |
| `ROUTER_TENANTS` | — | `key:name[:tpm],…` (keys are stored hashed) |
| `ROUTER_PRIMARY_TIMEOUT_MS` | `3000` | total deadline for the primary |
| `ROUTER_REDACT_PII` | `0` | run the Task 3 guard inside the router |
| `MCP_VALIDATION_ERRORS` | `protocol` | `tool` = SEP-1303 behaviour |

## Engineering

- **CI:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs ruff (including the no-`print` rule), the full suite on Python 3.12 and 3.13, the demo, a container build and a `compose config` check.
- **Containers:** one non-root image runs every service. The compose file puts the MCP server, the mock LLM and the providers on an
  internal-only network, and publishes only the gateways (and the primary's `/_mock` control port, so reviewers can trigger failover).
- **No real credentials anywhere.** The "secrets" in the tests are fake strings used to prove nothing leaks.

---

## Where this differs from popular prior art

| Concern | Common approach | This submission |
|---|---|---|
| Streaming PII | LiteLLM/Presidio buffers the **whole** stream, then masks; on error it **yields raw chunks** | exact incremental holdback; fails closed |
| Stream windows | NeMo `chunk_size`/`context_size` (`stream_first` emits *before* checking); fixed look-back windows that can leak once exceeded | minimal provable holdback (≈1 word), checked before emitting |
| Fields redacted | `content` only, or tool calls dropped | **every string** in every choice, plus MCP tool results; logprobs stripped |
| Token limits | charged after the response (Kong, Envoy), or trusting client-declared counts | reserved up front from our own estimate, reconciled to provider usage |
| Timeouts | per-read timeouts, or a deadline that ends when headers arrive | total deadline; TTFT commit for streams |
| MCP authz | static tokens; case-sensitive `startswith("admin_")` on raw bytes | JWT; normalised names; parser-differential, header and session defences |

## Iteration 3: what the adversarial review found

Reviewers given only the code and the brief wrote and ran exploit scripts. Every confirmed finding is fixed and has a regression test.

| Sev | Component | Finding | Fix |
|---|---|---|---|
| High | MCP gateway | `Mcp-Session-Id: decoy` + `Mcp-Session-Id: <victim>`: the first header was checked, the last forwarded | repeated headers rejected; HMAC-signed session ids |
| High | MCP server | `"id":"\ud800"` (lone surrogate) crashed the server | `-32600`; serialisation guarded |
| High | MCP server | the 4 MB line cap was checked only *after* buffering the line (300 MB → +312 MB RSS) | bounded `readline`, overflow drained in chunks |
| High | PII guardrail | PII leaked via legacy `function_call`, list `content`, `audio.transcript` | generic every-string walker |
| High | Router | invalid bytes before the first token → 500, no failover, stuck reservation | any pre-commit failure fails over; catch-all refund |
| High | Router | `n=10` reserved 1× `max_tokens`, so the window reached 80k/50k | reserve `n × max_tokens` |
| Med | MCP server | `"id": true / 1.0 / {} / null` silently dropped | `-32600` |
| Med | MCP gateway | session binding in memory: lost on restart, unknown ids allowed | stateless HMAC binding, fail-closed |
| Med | MCP gateway | 200 GET streams by one viewer starved admin POSTs | separate stream pool + per-subject cap |
| Med | Router | body parsed before auth, with no size cap | auth first; 413 |
| Med | Router | an in-stream error before the first token was "committed" | fails over |
| Med | PII guardrail | parallel tool calls without `index` shared a buffer | keyed by `index` / `id` / position |
| Med | PII guardrail | `SSN-123-45-6789`, dotted/dashed/mixed separators, IDN emails, full-width digits missed | patterns widened; Unicode Luhn |
| Low | various | zero-width reasons; gzip header stripped; batch header checks; 5000-digit ints; bad log level; `fallback` flag; Set-Cookie merging; spoofable request id; no half-open probe; evicted rows forgiven | all fixed |

**Iteration 4 (peer benchmark) added:**
- `a@b.com5` and grouped cards/SSNs glued to digits used to pass through. Fixed, plus the ground-truth oracle test that then caught the card-overlap bug.
- An import-time `print()` in a dependency could reach the wire. Fixed with the stdlib-only bootstrap and a negative control.
- A `regex` module quirk: an alternation with a variable-length lookbehind reports spurious partial matches. The bounded-memory test caught it, and the labelled-SSN rule now avoids lookbehinds.

## Known limitations / next steps
- **Task 3 is regex-based.** That's deterministic and fast.
  - Names, addresses and non-US IDs would need an NER pass (e.g. Presidio on each *released* segment; the holdback design makes that a drop-in).
  - Obfuscated forms (`jane at example dot com`) aren't caught.
  - Unseparated card numbers glued to other digits are deliberately treated as ids: a 19-digit id has about a 50% chance of containing a Luhn-valid prefix.
  - Prompt-side redaction isn't enabled.
- **Task 4 token estimates are heuristic** (chars/4 plus framing), and reconciliation corrects them; `tiktoken` would give exact pre-counts.
  Breaker state is per process (the limiter is shared via SQLite). At very high QPS a sliding-window counter or GCRA would
  trade exactness for O(1) state.
- **Task 2:** JWTs are HS256. Production would use IdP JWKS (RS256/ES256) and OAuth 2.1 resource metadata.
  MCP `tasks/*` results are not bound to a subject by the gateway. Tool pins live in memory, per process.
