"""Task 1's server over Streamable HTTP, so it can sit behind the Task 2 gateway.

    uv run uvicorn --factory task1_mcp_server.http_app:app_factory --port 9101
    # then point the gateway at it: GATEWAY_DOWNSTREAM_URL=http://127.0.0.1:9101/mcp

Same tools, same strict validation and error mapping as the stdio server (one ``build_server()``).
"""

from __future__ import annotations

from starlette.applications import Starlette

from .server import build_server


def app_factory(*, json_response: bool = False) -> Starlette:
    return build_server().streamable_http_app(json_response=json_response)
