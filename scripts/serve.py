"""Bring up a task's whole stack on fixed ports and print copy-paste commands to poke it by hand.

    uv run python scripts/serve.py task2    # Task 1 server (HTTP) + Task 2 gateway, tokens minted
    uv run python scripts/serve.py task3    # mock LLM + streaming PII guardrail
    uv run python scripts/serve.py task4    # two mock providers + rate-limit/failover router (+ PII guard)
    uv run python scripts/serve.py all

Ctrl-C to stop. Ports: 9101 task1-http, 9000 mcp-gateway, 8301 mock-llm, 8300 guardrail,
8401/8402 providers, 8400 router.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
import tempfile
import threading
import time
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def start(app, port: int) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.02)
    return server


def task2() -> None:
    from task1_mcp_server.http_app import app_factory as task1_http
    from task2_mcp_gateway.auth import TokenVerifier
    from task2_mcp_gateway.gateway import GatewaySettings, create_app

    secret = os.environ.get("GATEWAY_JWT_SECRET") or secrets.token_hex(32)
    start(task1_http(json_response=True), 9101)
    start(create_app(GatewaySettings("http://127.0.0.1:9101/mcp", secret, redact_tool_results="viewers")), 9000)
    v = TokenVerifier(secret)
    viewer, admin = v.mint("bob", "viewer"), v.mint("alice", "admin")
    hdr = "-H 'content-type: application/json' -H 'accept: application/json, text/event-stream'"
    print(f"""
Task 2 — MCP gateway on :9000 → Task 1 MCP server (Streamable HTTP) on :9101
  export GATEWAY_JWT_SECRET={secret}
  export VIEWER={viewer}
  export ADMIN={admin}
  (more tokens: uv run python -m task2_mcp_gateway.devtoken --role viewer --sub carol)

  # 1. start a session (copy the mcp-session-id response header into $SID)
  curl -si localhost:9000/mcp {hdr} -H "authorization: Bearer $VIEWER" \\
    -d '{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"2025-11-25","capabilities":{{}},"clientInfo":{{"name":"curl","version":"1"}}}}}}'
  # 2. viewer calls an admin tool → -32001, downstream never contacted
  curl -s localhost:9000/mcp {hdr} -H "authorization: Bearer $VIEWER" -H "mcp-session-id: $SID" \\
    -d '{{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{{"name":"admin_reset_key","arguments":{{}}}}}}'
  # 3. viewer reads a record → email redacted by the gateway's DLP (admins see it in clear)
  curl -s localhost:9000/mcp {hdr} -H "authorization: Bearer $VIEWER" -H "mcp-session-id: $SID" \\
    -d '{{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{{"name":"get_customer_record","arguments":{{"customer_id":"CUST-10001"}}}}}}'
""")


def task3() -> None:
    from task3_llm_guardrail.gateway import GuardSettings, create_app
    from task3_llm_guardrail.mock_llm import build

    start(build()[0], 8301)
    start(create_app(GuardSettings("http://127.0.0.1:8301/v1")), 8300)
    print("""
Task 3 — streaming PII guardrail on :8300 → mock OpenAI-compatible provider on :8301
  curl -sN localhost:8300/v1/chat/completions -H 'content-type: application/json' \\
    -d '{"stream":true,"mock_chunk_max":3,"mock_delay_ms":30,"messages":[{"role":"user",
         "content":"Email jane.doe@example.com, card 4111 1111 1111 1111, SSN 123-45-6789."}]}'
  (point at a real provider: GUARD_UPSTREAM_BASE_URL=https://api.openai.com/v1 GUARD_UPSTREAM_API_KEY=sk-...
   uv run uvicorn --factory task3_llm_guardrail.gateway:app_factory --port 8300)
""")


def task4() -> None:
    from task4_model_router.app import RouterSettings, create_app
    from task4_model_router.mock_providers import build
    from task4_model_router.router import Provider

    start(build("primary")[0], 8401)
    start(build("secondary")[0], 8402)
    db = Path(tempfile.gettempdir()) / "quilr-router-demo.db"
    settings = RouterSettings(
        db_path=str(db),
        primary=Provider("primary", "http://127.0.0.1:8401/v1"),
        secondary=Provider("secondary", "http://127.0.0.1:8402/v1", timeout_s=10),
        tenants=[("tenant-key-1", "acme", None), ("tenant-key-tiny", "tiny", 3000)],
        redact_pii=True,
    )
    start(create_app(settings), 8400)
    print(f"""
Task 4 — rate-limit + failover router on :8400 (SQLite: {db}) → providers :8401 (primary), :8402 (backup)
  chat() {{ curl -s -i localhost:8400/v1/chat/completions -H "authorization: Bearer ${{KEY:-tenant-key-1}}" \\
    -H 'content-type: application/json' -d "{{\\"max_tokens\\":${{MAX:-100}},\\"messages\\":[{{\\"role\\":\\"user\\",\\"content\\":\\"hi\\"}}]}}" \\
    | grep -iE '^HTTP|x-gateway|x-ratelimit|retry-after|^{{'; }}
  chat                                                                  # served by primary
  curl -s -XPOST localhost:8401/_mock/mode -d '{{"mode":"rate_limited"}}'; chat  # 429 → backup
  curl -s -XPOST localhost:8401/_mock/mode -d '{{"mode":"slow","delay_s":5}}'; chat  # cooldown → straight to backup
  KEY=tenant-key-tiny MAX=2900 chat; KEY=tenant-key-tiny MAX=2900 chat   # 2nd → 429 + Retry-After
  curl -s localhost:8401/_mock/stats                                    # how often the primary was actually hit
""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["task2", "task3", "task4", "all"])
    which = ap.parse_args().which
    for name, fn in (("task2", task2), ("task3", task3), ("task4", task4)):
        if which in (name, "all"):
            fn()
    print("Running. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
