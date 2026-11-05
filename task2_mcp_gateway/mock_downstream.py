"""Mock downstream MCP server (real MCP Streamable HTTP, built on the official SDK).

Exposes regular and ``admin_*`` tools and records every tools/call it executes at
``GET /_calls`` so tests can prove denied calls never reached it.

Run: uv run uvicorn --factory task2_mcp_gateway.mock_downstream:app_factory --port 9001
"""

from __future__ import annotations

import json
from typing import Any

import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.shared.exceptions import MCPError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

TOOLS = [
    types.Tool(
        name="get_customer_record",
        description="Read a customer record.",
        input_schema={"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]},
    ),
    types.Tool(name="list_open_tickets", description="List open support tickets.", input_schema={"type": "object"}),
    types.Tool(
        name="admin_reset_key",
        description="Rotate a tenant's API key (admin only).",
        input_schema={"type": "object", "properties": {"tenant": {"type": "string"}}, "required": ["tenant"]},
    ),
    types.Tool(name="admin_delete_user", description="Delete a user (admin only).", input_schema={"type": "object"}),
]


def build(*, json_response: bool = False) -> tuple[Starlette, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    known = {t.name for t in TOOLS}
    # Mutable so tests/demos can simulate a downstream silently changing a tool ("rug pull").
    tools_state: dict[str, Any] = {"tools": list(TOOLS)}

    async def list_tools(ctx: Any, params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools_state["tools"])

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        if params.name not in known:
            raise MCPError(code=types.INVALID_PARAMS, message=f"Unknown tool: {params.name}")
        calls.append({"name": params.name, "arguments": params.arguments or {}})
        payload = {"tool": params.name, "ok": True, "arguments": params.arguments or {}}
        if params.name == "get_customer_record":
            payload["record"] = {"name": "Ada Lovelace", "email": "ada@example.com", "card": "4111 1111 1111 1111"}
        return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(payload))])

    server = Server("mock-downstream", version="1.0.0", on_list_tools=list_tools, on_call_tool=call_tool)

    async def calls_endpoint(_: Request) -> JSONResponse:
        return JSONResponse(calls)

    app = server.streamable_http_app(
        json_response=json_response,
        custom_starlette_routes=[Route("/_calls", calls_endpoint, methods=["GET"])],
    )
    app.state.tools_state = tools_state
    return app, calls


def app_factory() -> Starlette:
    return build()[0]
