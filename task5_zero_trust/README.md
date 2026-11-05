# Task 5: Debugging a broken MCP connection in a corporate network

> **Problem statement (verbatim):** "An enterprise client deploys our LLM Gateway and MCP Gateway inside a
> zero-trust corporate network behind a strict egress proxy. Agents experience intermittent connection
> drops during multi-turn tool execution, resulting in hanging requests." Write out a step-by-step technical
> isolation and debugging strategy covering:
> 1. commands and diagnostic tools,
> 2. how to tell whether the failure is at the mTLS handshake, an SSE session timeout or LLM context truncation,
> 3. how to redesign the transport or connection pooling for resilience.
>
> Some copies of the brief only list this task as a focus area ("troubleshooting zero-trust network
> deployments"), so it is answered here for completeness.

This playbook uses the components in this repo (Task 2 = MCP gateway, Task 3/4 = LLM gateway). Every
header, log field and setting it mentions exists in the code. The main hypothesis (idle-timeout middleboxes
killing quiet SSE streams) is **reproduced and fixed in a test**:
[`tests/test_task5_idle_proxy.py`](../tests/test_task5_idle_proxy.py), using [`idle_proxy.py`](idle_proxy.py).

---

## 0. The first 15 minutes (high-visibility incident)

1. **Stabilise.** Ask the customer to set `GUARD_HEARTBEAT_S=10` / `ROUTER_HEARTBEAT_S=10` (or confirm they are on),
   and ask whether the proxy team can raise the idle timeout or add a TLS-inspection bypass for the two gateway FQDNs.
   Both are reversible and cheap, so they buy time while the root cause is found.
2. **Collect three failing request ids.** Every gateway response carries `x-request-id`, the same id is forwarded
   downstream, and every audit line includes it. Without ids there is no correlation, so this comes before any packet capture.
3. **Agree a cadence:** updates every 30 minutes. Each one covers the current hypothesis, the evidence for it and the next step. No guessing out loud.
4. **Define "hang" precisely:** how long, at which turn, which tool, whether it streams or not, and whether it is every user or one site / one egress node.
   "Intermittent during multi-turn tool execution" already points somewhere: the
   failures happen during the **longest quiet periods** (tool runs) and **late in conversations** (largest contexts).

## 1. Where it can break

```
agent ──TLS──► [egress proxy: CONNECT / TLS inspection / idle timeout] ──► LLM gateway ──► provider
   │                                                                  └──► MCP gateway ──► MCP server
   └── long-lived SSE responses (tokens, tool progress) + pooled keep-alive connections
```

| Hop | Typical failure | Looks like |
|---|---|---|
| Proxy CONNECT / TLS inspection | proxy re-signs certs; mTLS client cert stripped | failure at *connect*, `CERTIFICATE_VERIFY_FAILED`, TLS alert 116 `certificate_required` |
| Proxy / LB / NAT idle timer | silent SSE stream dropped after N s | stream stops mid-way; RST/FIN from a middlebox at a *constant* idle age |
| Proxy buffering | SSE held until the response ends | `ttfb ≈ total`; long tool runs look "hung", then arrive all at once |
| Keep-alive pool | reuses a connection the proxy already dropped | first request after an idle period hangs until a read timeout |
| LLM context | multi-turn history exceeds the window | HTTP 200 with `finish_reason: "length"` or a provider 400; truncated tool-call JSON makes the agent wait forever |

## 2. Commands (on-site / live session)

**Correlate first, using our own telemetry**
```bash
# MCP gateway audit trail for one request (JSON lines on stderr / journald)
journalctl -u mcp-gateway --since -1h -o cat | jq -c 'select(.rid=="<x-request-id>")'
# LLM router: per-provider outcome and latency for the same request
journalctl -u model-router -o cat | grep '<x-request-id>'   # attempts=[('primary','timeout',3001.2),...]
```

**Reproduce outside the agent, through the same proxy, with phase timings**
```bash
export HTTPS_PROXY=http://egress.corp:3128
curl -sS -o /dev/null --proxy "$HTTPS_PROXY" -w \
 'dns=%{time_namelookup} tcp=%{time_connect} tls=%{time_appconnect} ttfb=%{time_starttransfer} total=%{time_total} code=%{http_code}\n' \
 https://llm-gw.corp/healthz
# Streaming, timestamped per line: see exactly when the stream stops (moreutils `ts`)
curl -sN --proxy "$HTTPS_PROXY" https://llm-gw.corp/v1/chat/completions -H 'content-type: application/json' \
 -d '{"stream":true,"messages":[{"role":"user","content":"count slowly to 50"}]}' | ts '%.s'
env | grep -i _proxy; echo "NO_PROXY=$NO_PROXY"     # is the gateway even meant to go through the proxy?
```

**TLS / mTLS**
```bash
openssl s_client -connect llm-gw.corp:443 -servername llm-gw.corp -proxy egress.corp:3128 \
  -cert client.pem -key client.key -CAfile corp-ca.pem -showcerts -status </dev/null 2>&1 | \
  grep -E 'subject=|issuer=|Verify return|alert|Protocol|ALPN'
#  issuer = the corporate CA instead of ours  -> TLS inspection is re-signing (it breaks mTLS and pinning)
openssl x509 -in client.pem -noout -subject -issuer -dates      # expiry / rotation gaps
for ip in $(dig +short llm-gw.corp); do echo -n "$ip "; openssl s_client -connect $ip:443 \
  -servername llm-gw.corp </dev/null 2>&1 | grep -m1 'Verify return'; done                # one bad replica?
```

**Packets: who closes the connection, and after how long?**
```bash
tcpdump -i any -nn -s0 -w drop.pcap 'host 10.20.0.15 and tcp port 443'
tshark -r drop.pcap -Y 'tcp.flags.reset==1 || tcp.flags.fin==1' \
  -T fields -e frame.time_relative -e ip.src -e ip.dst -e tcp.stream -e tcp.flags.str
tshark -r drop.pcap -q -z conv,tcp                       # duration per stream: identical durations = a timer
tshark -r drop.pcap -Y 'tls.alert_message || tls.handshake.type==1'   # alerts / new handshakes
```

**Sockets, NAT and keepalive on the client / gateway hosts**
```bash
ss -tinpo state established '( dport = :443 or sport = :443 )'   # timers, retrans, keepalive, Send-Q
ss -s
sysctl net.ipv4.tcp_keepalive_time net.ipv4.tcp_keepalive_intvl net.netfilter.nf_conntrack_tcp_timeout_established
conntrack -L -p tcp --dport 443 2>/dev/null | head
```

**Inside a hung process**
```bash
py-spy dump --pid $(pgrep -f task4_model_router)   # where is the coroutine waiting? read? pool acquire? sqlite?
```

## 3. Finding the failure point: mTLS vs SSE timeout vs context truncation

Run the three checks in order of cost. Each has a signature that the others don't share.

| | mTLS handshake | SSE session / idle timeout | LLM context truncation |
|---|---|---|---|
| **When it fails** | establishing a *new* connection | mid-stream, after a quiet gap | late turns of long conversations |
| **Network evidence** | TLS alert or no ServerHello; `time_appconnect` missing; wrong issuer | RST/FIN from a middlebox IP at a **constant idle age** (60/120/300 s) | none: clean HTTP 200 |
| **Gateway evidence** | no request ever reaches the gateway (no `rid` in logs) | gateway logs show the stream *started*; Task 3 logs `stream closed … chunks_out=N` early | `finish_reason: "length"`, provider 400 `context_length_exceeded`, prompt tokens near the window |
| **Why "intermittent"** | only when the pool opens a new connection, or one replica has an expired / rotated cert | only tool calls or thinking pauses longer than the timer | only conversations past N turns |
| **Decisive test** | `openssl s_client` loop per replica, through and around the proxy | synthetic stream with a controlled gap: fails at a gap ≥ T and passes at a gap < T; passes with heartbeats on | replay the same conversation one turn shorter: the failure goes away; check `usage.prompt_tokens` |

**SSE timeout, bisected.** Run a provider that stays silent for a controlled time (our mock:
`"mock_ttft_ms": 45000`) and vary the gap: 30 s passes and 60 s fails means a 60 s idle timer. Repeat with
`GUARD_HEARTBEAT_S=15`. If it now passes, the diagnosis is confirmed. The test suite does exactly this at
small scale: a 1 s idle proxy kills the stream when heartbeats are off and lets it complete when they are on.

**Proxy buffering, a lookalike.** If `ttfb ≈ total` on a streaming call, the proxy is buffering, not
dropping. Our gateways send `X-Accel-Buffering: no` and `Cache-Control: no-cache` and never gzip SSE, but a
TLS-inspecting proxy can still buffer. Fix it with a bypass rule.

**Context truncation, confirmed.** Agents "hang" when a truncated stream cuts off a tool call's JSON
`arguments`: the agent waits for a tool call that can never be parsed. Look for `finish_reason:"length"` on
the last chunk. The Task 4 router logs `reserved=… actual=…` tokens per request, which shows the
prompt growing turn by turn.

**The fourth suspect: stale pooled connections.** If only the *first* request after an idle period hangs,
the pool is reusing a socket the proxy already dropped. The pcap shows the request leaving with no response,
then a retransmission.

## 4. Redesign for resilience

**Keep long-lived streams alive and visible**
- **SSE heartbeats**: `: ping` comments while a stream is idle (implemented: `GUARD_HEARTBEAT_S`, `ROUTER_HEARTBEAT_S`,
  default 15 s, below common 30–60 s idle timers). They are read on a separate task, so the timer never
  cancels an in-flight socket read. Cleanup is shielded, so a client disconnect still cancels upstream generation.
- **Anti-buffering headers** on every SSE response, and no compression middleware on stream routes (implemented).
- **Fail fast, never hang**:
  - a total deadline per provider attempt (3000 ms) and a time-to-first-token commit for streams (implemented);
  - a stall timeout between chunks;
  - an explicit in-stream error event instead of a silent stall (implemented: `upstream_timeout`, `upstream_stream_interrupted`).

**Connection pooling**
- Keep the pool's keep-alive expiry *below* the smallest middlebox idle timer, so pooled sockets are retired
  before the proxy kills them. Enable TCP keepalive (`TCP_KEEPIDLE` < NAT timeout) on long-lived upstream sockets.
- **Separate pools** for long-lived streams and short RPCs, with per-subject stream caps, so one slow class
  can't starve the other (implemented in the MCP gateway).
- Retry once on a stale-connection error, but **only for idempotent calls**. Side-effecting tools carry
  idempotency keys (implemented: `trigger_refund(idempotency_key=…)`), so an agent's retry after a drop can't
  refund twice.
- A circuit breaker with a **half-open single probe** (implemented), so a flapping provider or proxy path isn't
  hammered by every request.

**Stop holding connections across long tool runs**
- Make long-running tools **asynchronous**. MCP 2025-11-25 *tasks* let a `tools/call` return a task handle at once,
  and the agent then polls `tasks/get`. No connection has to survive a 5-minute tool. This is the most effective
  structural fix for "drops during multi-turn tool execution".
- Where streams remain, make them **resumable**. Streamable HTTP SSE event ids plus `Last-Event-ID` let the client
  reconnect and replay (the gateway forwards `Last-Event-ID`; the SDK server takes an `event_store`).
  The 2026-07-28 revision is stateless per request, so a dropped request is simply retried, and the gateway's
  HMAC-bound session ids survive gateway restarts and replicas.
- Transport: HTTP/2 through the proxy where it supports it (fewer connections, PING frames act as keepalive).
  WebSockets are usually *worse* through inspecting proxies, so avoid them.

**mTLS in a zero-trust network**
- Terminate mTLS at the workload (mesh sidecar / SPIFFE SVIDs) and **bypass TLS inspection** for gateway FQDNs.
  Re-signing proxies and client-certificate auth don't mix.
- Rotate certificates with overlap, and alert at 30 / 7 / 1 days before expiry. Add a synthetic check per replica.

**Context truncation**
- Budget the context at the gateway: the router already estimates prompt tokens before every call.
  Extend that into a per-model context guard that summarises or trims history, or returns an explicit
  `400 context_length_exceeded` rather than letting a provider truncate silently.
- Treat `finish_reason: "length"` on a stream that was emitting a tool call as an error event, never as success.

**Observability, so the next incident takes 10 minutes**
- Propagate `x-request-id` end to end (implemented) and record per-hop latency: proxy, gateway, provider.
- Run a **canary** every 10 minutes from inside the customer network. It opens a stream that idles for 5 minutes,
  so idle-timeout regressions are caught before users notice.
- Add dashboards for stream duration histograms. A spike at a single duration means a timer.

## 5. Where this lives in the code

| Remediation | Implementation |
|---|---|
| SSE heartbeats (read on a separate task, shielded cleanup) | [`task3_llm_guardrail/sse.py`](../task3_llm_guardrail/sse.py) `with_heartbeat` · used by the Task 3 gateway and the Task 4 router |
| Idle-timeout middlebox reproduction | [`idle_proxy.py`](idle_proxy.py), [`tests/test_task5_idle_proxy.py`](../tests/test_task5_idle_proxy.py) |
| Total deadline + TTFT commit + in-stream errors | [`task4_model_router/router.py`](../task4_model_router/router.py) |
| Half-open breaker, cooldown on 429 | `CircuitBreaker` in the same file |
| Separate stream pool + per-subject cap | [`task2_mcp_gateway/gateway.py`](../task2_mcp_gateway/gateway.py) `stream_client`, `max_streams_per_subject` |
| Restart-proof session binding | [`task2_mcp_gateway/sessions.py`](../task2_mcp_gateway/sessions.py) |
| Idempotent side-effecting tool | `trigger_refund(idempotency_key=…)` in [`task1_mcp_server`](../task1_mcp_server) |
| Request-id propagation + JSON audit log | both gateways (`x-request-id`, `task2.gateway.audit`) |
