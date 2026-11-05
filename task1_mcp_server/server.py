"""Task 1 - Customer-support MCP server over stdio with strict input validation.

Run:   uv run python -m task1_mcp_server.server

STDIO contract
--------------
* fd 1 (stdout) carries *only* newline-delimited JSON-RPC frames.
* Everything else - logging, warnings, stray ``print()``, output of C extensions
  or child processes - goes to fd 2 (stderr).

This is enforced at the file-descriptor level in :func:`_claim_stdio`, not just by
"remembering not to print": the real stdout is duplicated onto a private fd that
only the transport writer owns, and fd 1 is then re-pointed at stderr.

Error mapping
-------------
* Malformed JSON on the wire                  -> -32700 Parse error (transport.py)
* Valid JSON but not a JSON-RPC message       -> -32600 Invalid Request (transport.py)
* Unknown method                              -> -32601 Method not found (SDK)
* Unknown tool name / schema-invalid arguments -> -32602 Invalid params
* Unexpected exception inside a tool          -> -32603 Internal error (sanitised)
* Business-rule failure (customer not found, refund exceeds balance)
  -> a normal result with ``isError: true`` so the calling model can see and react
  to it, as the MCP spec prescribes for tool *execution* errors.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, BinaryIO

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.shared.exceptions import MCPError
from pydantic import ValidationError

from .data import CustomerStore, default_store
from .schemas import (
    TOOL_INPUT_MODELS,
    TOOL_OUTPUT_MODELS,
    CustomerRecord,
    GetCustomerRecordInput,
    RefundReceipt,
    TriggerRefundInput,
)
from .transport import jsonrpc_stdio

logger = logging.getLogger("task1.mcp_server")

SERVER_NAME = "customer-support-mcp"
SERVER_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# STDIO isolation
# --------------------------------------------------------------------------- #
def _configure_logging() -> None:
    level = os.environ.get("MCP_LOG_LEVEL", "INFO").upper()
    if level not in logging.getLevelNamesMapping():
        level = "INFO"  # a typo in config must not take the server down
    logging.basicConfig(
        level=level,
        stream=sys.stderr,  # never stdout
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.captureWarnings(True)


def _claim_stdio() -> tuple[BinaryIO, BinaryIO]:
    """Reserve the real stdout for JSON-RPC and divert everything else to stderr.

    Returns binary handles for the wire (stdin, stdout).
    """
    sys.stdout.flush()
    # Private duplicate of the real stdout; os.dup() fds are non-inheritable, so
    # child processes cannot write onto the wire either.
    wire_out_fd = os.dup(1)
    # fd 1 now *is* stderr: print(), C-level printf(), subprocess output all land there.
    os.dup2(2, 1)
    sys.stdout = sys.stderr  # Python-level belt-and-braces

    wire_in_fd = os.dup(0)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)  # nothing else may consume bytes meant for the JSON-RPC reader
    os.close(devnull)
    return os.fdopen(wire_in_fd, "rb"), os.fdopen(wire_out_fd, "wb")


# --------------------------------------------------------------------------- #
# Tool definitions
# --------------------------------------------------------------------------- #
_TOOL_META: dict[str, dict[str, Any]] = {
    "get_customer_record": {
        "title": "Get customer record",
        "description": "Fetch a customer's profile and refundable balance by customer ID (CUST-XXXXX).",
        "annotations": types.ToolAnnotations(
            read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
        ),
    },
    "trigger_refund": {
        "title": "Trigger refund",
        "description": (
            "Issue a refund to a customer. Amount must be positive (max 2 decimals) and not exceed the "
            "customer's refundable balance; reason must be at least 10 characters."
        ),
        # Moves money: clients should ask for human confirmation before calling.
        "annotations": types.ToolAnnotations(
            read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=False
        ),
    },
}


def _schema(model: type) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    assert "$defs" not in schema, "keep tool schemas flat ($ref-free) for weaker clients"
    return schema


def _tool_definitions() -> list[types.Tool]:
    # Stable order (dict insertion order) - the 2026-07-28 spec asks for deterministic tools/list output.
    return [
        types.Tool(
            name=name,
            input_schema=_schema(model),
            output_schema=_schema(TOOL_OUTPUT_MODELS[name]),
            **_TOOL_META[name],
        )
        for name, model in TOOL_INPUT_MODELS.items()
    ]


def _validation_error_data(tool: str, exc: ValidationError) -> dict[str, Any]:
    """Structured, client-safe error detail. Never echoes the offending input values back."""
    return {
        "tool": tool,
        "errors": [
            {
                "field": ".".join(str(p) for p in err["loc"]) or "<arguments>",
                "type": err["type"],
                "message": err["msg"],
            }
            for err in exc.errors(include_url=False, include_input=False)
        ],
    }


def _ok(model: Any) -> types.CallToolResult:
    payload = model.model_dump()  # validated against the advertised outputSchema on the way out
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload))],
        structured_content=payload,
        is_error=False,
    )


def _tool_error(message: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=message)], is_error=True)


def build_server(store: CustomerStore | None = None, *, validation_errors: str | None = None) -> Server:
    """``validation_errors``:
    * ``"protocol"`` (default, what this assessment asks for): schema-invalid arguments -> JSON-RPC -32602.
    * ``"tool"``: return them as ``isError: true`` results instead - what MCP 2025-11-25+ (SEP-1303)
      recommends so the calling model can read the error and self-correct.
    """
    store = store or default_store()
    validation_errors = validation_errors or os.environ.get("MCP_VALIDATION_ERRORS", "protocol")
    if validation_errors not in {"protocol", "tool"}:
        raise ValueError("validation_errors must be 'protocol' or 'tool'")
    tool_defs = _tool_definitions()

    async def list_tools(ctx: Any, params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tool_defs)

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        name = params.name
        model = TOOL_INPUT_MODELS.get(name)
        if model is None:
            raise MCPError(code=types.INVALID_PARAMS, message=f"Unknown tool: {name}", data={"tool": name})

        try:
            args = model.model_validate(params.arguments or {})
        except ValidationError as exc:
            logger.info("rejected %s call: %d validation error(s)", name, exc.error_count())
            data = _validation_error_data(name, exc)
            if validation_errors == "tool":
                result = _tool_error(f"Invalid arguments for tool '{name}': " + json.dumps(data["errors"]))
                result.structured_content = data
                return result
            raise MCPError(
                code=types.INVALID_PARAMS,
                message=f"Invalid arguments for tool '{name}'",
                data=data,
            ) from None

        try:
            if isinstance(args, GetCustomerRecordInput):
                customer = store.get(args.customer_id)
                if customer is None:
                    return _tool_error(f"Customer {args.customer_id} not found")
                return _ok(CustomerRecord(**customer.to_public_dict()))

            if isinstance(args, TriggerRefundInput):
                try:
                    refund_id, customer = store.refund(args.customer_id, args.amount)
                except (LookupError, ValueError) as exc:
                    return _tool_error(str(exc))
                logger.info("refund %s issued for %s amount=%.2f", refund_id, customer.customer_id, args.amount)
                return _ok(
                    RefundReceipt(
                        refund_id=refund_id,
                        customer_id=customer.customer_id,
                        amount=args.amount,
                        reason=args.reason,
                        status="processed",
                        remaining_refundable_balance=customer.refundable_balance,
                    )
                )
        except Exception:
            # Full traceback to stderr only; the client gets a generic, non-leaky error.
            logger.exception("unhandled error in tool %s", name)
            raise MCPError(code=types.INTERNAL_ERROR, message="Internal error while executing tool") from None

        raise MCPError(code=types.INTERNAL_ERROR, message="Tool has no handler")  # pragma: no cover

    return Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        instructions="Customer support tools: look up customer records and issue refunds.",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


async def _serve() -> None:
    wire_in, wire_out = _claim_stdio()
    server = build_server()
    logger.info("%s v%s listening on stdio", SERVER_NAME, SERVER_VERSION)
    async with jsonrpc_stdio(wire_in, wire_out) as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
    logger.info("stdin closed, shutting down")


def main() -> None:
    _configure_logging()
    try:
        anyio.run(_serve)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
