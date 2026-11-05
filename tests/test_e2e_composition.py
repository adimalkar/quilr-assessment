"""End-to-end composition: the four tasks as one system.

SDK client ──► Task 2 gateway ──► Task 1 MCP server (Streamable HTTP)
OpenAI SDK-style client ──► Task 4 router (+ Task 3 redaction) ──► providers
"""

from __future__ import annotations

import httpx2
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError

from task1_mcp_server.http_app import app_factory as task1_http
from task2_mcp_gateway.auth import TokenVerifier
from task2_mcp_gateway.gateway import GatewaySettings
from task2_mcp_gateway.gateway import create_app as gateway_app

from .conftest import run_uvicorn

SECRET = "e2e-secret-e2e-secret-e2e-secret-e2e!"


@pytest.fixture(scope="module")
def mcp_stack():
    with run_uvicorn(task1_http()) as task1:
        settings = GatewaySettings(downstream_url=f"{task1}/mcp", jwt_secret=SECRET, redact_tool_results="viewers")
        with run_uvicorn(gateway_app(settings)) as gw:
            yield gw


async def test_task1_server_behind_task2_gateway(mcp_stack):
    v = TokenVerifier(SECRET)
    async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {v.mint('bob', 'viewer')}"}, timeout=10) as http:
        async with streamable_http_client(f"{mcp_stack}/mcp", http_client=http) as (read, write, *_):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {t.name for t in (await session.list_tools()).tools}
                assert tools == {"get_customer_record", "trigger_refund"}

                # Task 1's strict validation still surfaces as -32602 through the gateway
                with pytest.raises(MCPError) as exc:
                    await session.call_tool("get_customer_record", {"customer_id": "nope"})
                assert exc.value.code == -32602

                # Task 2's DLP redacts the customer's email for a viewer (Task 3's redactor)
                rec = await session.call_tool("get_customer_record", {"customer_id": "CUST-10001"})
                text = rec.content[0].text
                assert "ada@example.com" not in text and "[REDACTED]" in text and "Ada Lovelace" in text

                # and an unknown admin_* tool is stopped at the gateway, never reaching Task 1
                with pytest.raises(MCPError) as exc:
                    await session.call_tool("admin_reset_key", {})
                assert exc.value.code == -32001


async def test_admin_sees_clear_data_through_the_same_stack(mcp_stack):
    v = TokenVerifier(SECRET)
    async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {v.mint('alice', 'admin')}"}, timeout=10) as http:
        async with streamable_http_client(f"{mcp_stack}/mcp", http_client=http) as (read, write, *_):
            async with ClientSession(read, write) as session:
                await session.initialize()
                rec = await session.call_tool("get_customer_record", {"customer_id": "CUST-10001"})
                assert rec.structured_content["email"] == "ada@example.com"
