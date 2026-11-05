# Rubric map: brief → code → proving tests

Each evaluation criterion from the brief is quoted, then where it is implemented, then the tests that prove it
(`tests/<file>::<test>`). Run any line with `uv run pytest "tests/<file>::<test>"`.

## Task 1: MCP server with strict validation and transport handling

**Requirements:** an official-SDK server with `get_customer_record` (`CUST-XXXXX`) and `trigger_refund` (positive float,
reason ≥ 10 chars); Zod/Pydantic validation; invalid input → standard JSON-RPC errors; stdio transport;
stdout carries only JSON-RPC and logs go to stderr.

| Criterion | Where | Proven by (`test_task1_mcp_server.py`) |
|---|---|---|
| **STDIO isolation**: "stdout is pure JSON-RPC without accidental console.log or print" | fd-level claim in [`stdio_guard.py`](../task1_mcp_server/stdio_guard.py), run *before* third-party imports by [`__main__.py`](../task1_mcp_server/__main__.py); ruff `T20` bans `print` | `test_stray_prints_never_reach_stdout` (print, sys.stdout, `os.write(1)`, child process, logging); `test_import_time_prints_cannot_reach_the_wire`; **negative control** `test_negative_control_purity_check_can_fail`; every test asserts `assert_stdout_pure()` |
| **Protocol compliance**: "standard JSON-RPC error mapping and execution flow" | [`transport.py`](../task1_mcp_server/transport.py) (-32700/-32600), [`server.py`](../task1_mcp_server/server.py) (-32601/-32602/-32603, `isError` for business rules) | `test_parse_error_and_invalid_request_framing`, `test_invalid_utf8_is_parse_error`, `test_unknown_method`, `test_unknown_tool_is_invalid_params`, `test_invalid_request_ids_get_an_error_not_silence`, `test_malformed_notification_gets_no_reply`, `test_unknown_customer_is_tool_execution_error`, `test_official_sdk_client_interop`, `test_modern_protocol_discover_path` |
| **Validation**: "robust schema definitions and edge-case handling for malformed input fields" | strict Pydantic models in [`schemas.py`](../task1_mcp_server/schemas.py) → `inputSchema`/`outputSchema` | `test_customer_id_format_rejected` (13 cases), `test_refund_validation_rejected` (13 cases), `test_nan_and_infinity_rejected`, `test_missing_arguments_lists_all_required_fields`, `test_validation_errors_do_not_echo_input`, `test_reason_must_have_real_content`, `test_lone_surrogates_rejected_and_server_survives`, `test_oversized_frame_is_bounded_and_answered`, `test_huge_integer_is_invalid_request_not_parse_error` |
| Beyond the brief | idempotency keys; tool annotations; SEP-1303 toggle; HTTP mode | `test_idempotency_key_prevents_double_refund`, `test_tool_annotations_and_output_schema`, `test_validation_errors_as_tool_results_mode`, `test_e2e_composition.py::test_task1_server_behind_task2_gateway` |

## Task 2: MCP security gateway proxy (tool filtering and auth)

**Requirements:** read `Bearer <token>`, extract the role; forward `tools/list` transparently; on `tools/call`, check `params.name`.
An `admin_*` tool called without the admin role → `-32001 Unauthorized Tool Call`, and the downstream is never called.

| Criterion | Where | Proven by (`test_task2_mcp_gateway.py`) |
|---|---|---|
| **"Parsing JSON-RPC wire format structures correctly"** | strict parser in [`jsonrpc.py`](../task2_mcp_gateway/jsonrpc.py): dup keys, case-variant keys, surrogates, NaN, null ids, envelope checks | `test_parse_error`, `test_invalid_envelopes`, `test_duplicate_key_smuggling_rejected`, `test_case_variant_key_smuggling_rejected`, `test_unsafe_json_is_invalid_request_not_500`, `test_null_request_id_rejected`, `test_batches_rejected_and_never_forwarded`, `test_tools_call_without_valid_name` |
| **"Proxy middleware construction and HTTP request/response forwarding"** | [`gateway.py`](../task2_mcp_gateway/gateway.py): streamed byte-for-byte (JSON + SSE), RFC 9110 hop-by-hop handling, signed `Mcp-Session-Id`, GET/DELETE, separate stream pool | every test runs against **both** JSON and SSE downstreams; `test_official_sdk_client_through_gateway`, `test_response_headers_preserved_correctly`, `test_client_token_not_forwarded_downstream`, `test_get_streams_capped_per_subject`, `test_downstream_unavailable_is_sanitised`, `test_body_size_limit` |
| **"Fine-grained, method-level authorization logic and clean error handling"** | [`policy.py`](../task2_mcp_gateway/policy.py) (normalised names), [`auth.py`](../task2_mcp_gateway/auth.py) (JWT), [`sessions.py`](../task2_mcp_gateway/sessions.py) | `test_viewer_admin_tool_blocked_and_never_reaches_downstream`, `test_admin_can_call_admin_tool`, `test_lookalike_admin_names_blocked` (6 cases), `test_invalid_tokens_rejected` (6 cases incl. `alg:none`), `test_mirror_header_mismatch_rejected`, `test_session_bound_to_subject`, `test_duplicate_session_header_cannot_bypass_binding`, `test_tools_call_notification_never_forwarded`, `test_foreign_origin_rejected` |
| Beyond the brief | DLP on tool results, tool pinning (rug pull), list filter | `test_tool_result_dlp_for_viewers`, `test_tool_definition_rug_pull_quarantined`, `test_optional_tools_list_filter` |

## Task 3: LLM gateway streaming guardrail (PII redaction)

**Requirements:** intercept the stream in real time; redact emails, SSNs and cards; don't accumulate the full response; minimise TTFT.

| Criterion | Where | Proven by (`test_task3_guardrail.py`) |
|---|---|---|
| **"Efficient asynchronous stream chunking and buffer state management"** | [`sse.py`](../task3_llm_guardrail/sse.py) decoder, [`stream_guard.py`](../task3_llm_guardrail/stream_guard.py) per-(choice, JSON-path) buffers, [`gateway.py`](../task3_llm_guardrail/gateway.py) | `test_sse_crlf_split_and_multibyte_split` (every split point), `test_multiple_choices_isolated`, `test_parallel_tool_calls_without_index_do_not_mix`, `test_tool_call_arguments_are_redacted`, `test_unknown_and_nested_text_fields_redacted`, `test_client_disconnect_cancels_upstream`, `test_heartbeat_keeps_idle_stream_alive` |
| **"Performant string/regex pattern matching over partial text streams"** | [`redactor.py`](../task3_llm_guardrail/redactor.py): `regex` partial-match hold point plus one shared scan rule | `test_chunking_invariance_property` (300), **independent oracle** `test_ground_truth_oracle_no_injected_pii_survives` (250), `test_char_by_char_and_whole`, `test_detects`, `test_does_not_over_redact`, `test_detection_gaps_closed`, `test_digit_and_letter_adjacency`; [benchmarks](BENCHMARKS.md) §1 |
| **"Memory efficiency and low-latency proxying design"** | bounded patterns → buffer ≤ 640 chars; no GZip; `X-Accel-Buffering: no` | `test_bounded_memory_on_long_stream`, `test_holdback_is_minimal_and_streams_promptly`, `test_first_token_latency_not_degraded`; [benchmarks](BENCHMARKS.md) §2 |
| Fail closed / sanitised errors | gateway error paths | `test_upstream_error_sanitised`, `test_truncated_upstream_reported`, `test_upstream_auth_failure_is_gateway_error`, `test_logprobs_stripped` |

## Task 4: rate-limiting and model fallback router

**Requirements:** token-aware sliding window of 50k tokens/min per tenant key; on a 429 or a 3000 ms timeout → secondary provider;
standardised errors without upstream internals; on-disk SQLite.

| Criterion | Where | Proven by (`test_task4_router.py`) |
|---|---|---|
| **"Async concurrency handling and timeout race conditions"** | `asyncio.timeout` total deadline + TTFT commit in [`router.py`](../task4_model_router/router.py); `BEGIN IMMEDIATE` in [`limiter.py`](../task4_model_router/limiter.py) | `test_failover_on_timeout_at_3000ms` (3.0–3.8 s window), `test_total_deadline_not_per_read_timeout` (trickling provider), `test_stream_ttft_timeout_fails_over`, `test_thread_concurrency_never_overshoots`, **`test_multi_process_concurrency_never_overshoots`** (4 processes, 1 DB file), `test_concurrent_requests_respect_limit`; [benchmarks](BENCHMARKS.md) §3 |
| **"Accurate rate-limiter state eviction and token tracking logic"** | sliding log, reserve → reconcile → refund | `test_sliding_window_expiry_is_exact` (ms-exact), `test_no_fixed_window_boundary_burst`, `test_retry_after_walks_oldest_first`, `test_reconcile_returns_capacity_and_release_refunds`, `test_eviction_deletes_rows`, `test_reconcile_after_eviction_reinserts_usage`, `test_reservation_accounts_for_n`, `test_state_persists_on_disk` |
| **"Graceful fallback mechanics and standardized error sanitization"** | failover policy + half-open breaker; [`errors.py`](../task4_model_router/errors.py) | `test_failover_on_429`, `test_failover_on_5xx`, `test_failover_on_connection_refused`, `test_no_failover_on_400_and_body_not_leaked`, `test_cooldown_skips_rate_limited_primary`, `test_half_open_allows_single_probe`, `test_stream_failure_before_first_token_fails_over`, `test_all_providers_down_sanitised_and_refunded`, `test_validation_errors_standardised`, `test_malformed_json_and_unknown_route`, `test_sqlite_lock_is_fast_503_not_500` |
| **"Use on disk sqlite"** | WAL, `busy_timeout`, hashed keys | `test_state_persists_on_disk`, `test_api_keys_never_stored_in_plaintext`, `test_multi_process_concurrency_never_overshoots` |

## Task 5: zero-trust debugging scenario

| Criterion | Where |
|---|---|
| "Enterprise network debugging instincts and practical Linux CLI proficiency" | [`task5_zero_trust/README.md`](../task5_zero_trust/README.md) §2 |
| "Systematic troubleshooting workflow under high customer visibility" | §0 and §3 (discriminating signatures and decisive tests) |
| "Architectural clarity when proposing long-term technical remediations" | §4–5, with the core remediation **implemented and tested**: `tests/test_task5_idle_proxy.py::test_idle_timeout_middlebox_vs_heartbeats` |
