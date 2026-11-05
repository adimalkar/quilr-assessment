"""Minimal, strict JSON-RPC 2.0 wire helpers for the gateway."""

from __future__ import annotations

import base64
import json
from typing import Any

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
# Implementation-defined server error range (-32000..-32099)
AUTHENTICATION_REQUIRED = -32000
UNAUTHORIZED_TOOL_CALL = -32001
UPSTREAM_UNAVAILABLE = -32002
UPSTREAM_TIMEOUT = -32003
FORBIDDEN = -32004
# Spec-defined (MCP 2026-07-28): Mcp-Method / Mcp-Name header disagrees with the body
HEADER_MISMATCH = -32020

_RESERVED_TOP_LEVEL = {"jsonrpc", "id", "method", "params", "result", "error"}
_RESERVED_TOOL_CALL = {"name", "arguments", "_meta"}


class UnsafeJSONError(ValueError):
    """Syntactically valid JSON we still refuse (-32600): ambiguity or values that break later stages."""


class DuplicateKeyError(UnsafeJSONError):
    def __str__(self) -> str:
        return f"Duplicate JSON key: {self.args[0]}"


def _bounded_int(text: str) -> int:
    if len(text) > 4000:
        raise UnsafeJSONError("Number too large")
    return int(text)


def _has_lone_surrogate(obj: Any) -> bool:
    stack = [obj]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            if any("\ud800" <= ch <= "\udfff" for ch in item):
                return True
        elif isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return False


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            # {"name": "get_customer", "name": "admin_reset_key"} - different parsers keep the first or the
            # last key; refusing ambiguity removes any parser-differential between gateway and downstream.
            raise DuplicateKeyError(key)
        obj[key] = value
    return obj


def _no_constants(token: str) -> Any:
    raise ValueError(f"non-standard JSON constant {token}")


def loads_strict(raw: bytes) -> Any:
    obj = json.loads(raw, object_pairs_hook=_reject_duplicates, parse_constant=_no_constants, parse_int=_bounded_int)
    if _has_lone_surrogate(obj):
        # "\ud800" is legal JSON but not Unicode: downstream decoders disagree on how to treat it.
        raise UnsafeJSONError("Unpaired UTF-16 surrogate in string")
    return obj


def dumps(obj: Any) -> bytes:
    # ensure_ascii: output is pure ASCII, so it can never fail to encode or be re-interpreted downstream.
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=True).encode()


def valid_id(value: Any) -> bool:
    return value is None or isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))


def error(request_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": request_id if valid_id(request_id) else None, "error": err}


def _case_variant(keys: Any, reserved: set[str]) -> str | None:
    for key in keys:
        if key not in reserved and key.casefold() in reserved:
            return key
    return None


def smuggling_problem(msg: dict[str, Any]) -> str | None:
    """Reject keys that differ from a protocol key only by case (``"Name"``, ``"Method"``).

    Case-insensitive JSON decoders downstream (e.g. Go's encoding/json, CVE-2026-27896) would bind them to
    the real field, so the gateway's view of the call and the downstream's could diverge."""
    if bad := _case_variant(msg.keys(), _RESERVED_TOP_LEVEL):
        return f"ambiguous key {bad!r}"
    params = msg.get("params")
    if (
        msg.get("method") == "tools/call"
        and isinstance(params, dict)
        and (bad := _case_variant(params.keys(), _RESERVED_TOOL_CALL))
    ):
        return f"ambiguous key params.{bad}"
    return None


def decode_header_value(value: str) -> str:
    """Undo the MCP ``=?base64?...?=`` sentinel used for header values that aren't HTTP-safe."""
    if value.startswith("=?base64?") and value.endswith("?="):
        try:
            return base64.b64decode(value[9:-2], validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return "\x00<undecodable>"  # can never equal a real body value -> mismatch
    return value


def is_request(msg: dict[str, Any]) -> bool:
    return "method" in msg and "id" in msg


def is_notification(msg: dict[str, Any]) -> bool:
    return "method" in msg and "id" not in msg


def envelope_problem(msg: Any) -> str | None:
    """Return why ``msg`` is not a valid JSON-RPC 2.0 message, or None if it is."""
    if not isinstance(msg, dict):
        return "message must be a JSON object"
    if msg.get("jsonrpc") != "2.0":
        return "jsonrpc must be '2.0'"
    if "id" in msg and not valid_id(msg["id"]):
        return "id must be a string, integer or null"
    if "method" in msg and "id" in msg and msg["id"] is None:
        return "request id must not be null (MCP)"
    if "method" in msg:
        if not isinstance(msg["method"], str) or not msg["method"]:
            return "method must be a non-empty string"
        if "params" in msg and not isinstance(msg["params"], (dict, list)):
            return "params must be an object or array"
    elif not ("result" in msg or "error" in msg):
        # client -> server responses (e.g. to sampling/elicitation requests) are legitimate
        return "message must be a request, notification or response"
    return None
