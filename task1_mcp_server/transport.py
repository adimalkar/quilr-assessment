"""Newline-delimited JSON-RPC stdio transport with spec-compliant framing errors.

Functionally equivalent to ``mcp.server.stdio.stdio_server`` but, instead of
silently dropping undecodable lines, it answers them as JSON-RPC 2.0 requires:

* not valid JSON                         -> ``-32700 Parse error``  (id: null)
* valid JSON, not a JSON-RPC message     -> ``-32600 Invalid Request`` (id echoed when recoverable)
* JSON array (batch)                     -> ``-32600`` (batching was removed from MCP in 2025-06-18)
* invalid UTF-8                          -> ``-32700`` (the line is not decodable JSON text)
* a malformed *notification* (has ``method``, no ``id``) gets no reply - JSON-RPC forbids
  responding to notifications, even broken ones.
* stdout closed by the client (EPIPE)   -> log to stderr and shut down cleanly.

All writes to the wire - SDK responses and these framing errors - go through one
lock so frames can never interleave.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, BinaryIO

import anyio
import mcp_types as types
from mcp.shared._context_streams import create_context_streams
from mcp.shared.message import SessionMessage
from pydantic import ValidationError

logger = logging.getLogger("task1.transport")

MAX_LINE_BYTES = 4 * 1024 * 1024  # refuse absurd frames instead of buffering them forever


def _error_frame(code: int, message: str, request_id: Any = None) -> str:
    frame = types.JSONRPCError(jsonrpc="2.0", id=request_id, error=types.ErrorData(code=code, message=message))
    return frame.model_dump_json(by_alias=True, exclude_unset=True)


class _NumberTooLarge(ValueError):
    pass


def _bounded_int(text: str) -> int:
    if len(text) > 4000:  # CPython refuses > 4300 digits anyway; answer -32600 rather than -32700
        raise _NumberTooLarge(text[:20])
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


def _looks_like_notification(payload: Any) -> bool:
    return isinstance(payload, dict) and "method" in payload and "id" not in payload


def _recover_id(payload: Any) -> str | int | None:
    if isinstance(payload, dict):
        rid = payload.get("id")
        if isinstance(rid, str) or (isinstance(rid, int) and not isinstance(rid, bool)):
            return rid
    return None


@asynccontextmanager
async def jsonrpc_stdio(wire_in: BinaryIO, wire_out: BinaryIO) -> AsyncIterator[tuple[Any, Any]]:
    # Binary in: we decode each line strictly ourselves so invalid UTF-8 becomes a parse error
    # instead of being silently "repaired" with U+FFFD and executed.
    stdin = anyio.wrap_file(wire_in)
    stdout = anyio.wrap_file(io.TextIOWrapper(wire_out, encoding="utf-8", newline="\n"))
    write_lock = anyio.Lock()
    peer_gone = anyio.Event()

    async def emit(frame: str) -> None:
        # Frames are compact single-line JSON (pydantic/json never emit raw newlines inside strings).
        async with write_lock:
            if peer_gone.is_set():
                return
            try:
                await stdout.write(frame + "\n")
                await stdout.flush()
            except (BrokenPipeError, ConnectionResetError, ValueError):
                logger.warning("client closed stdout; shutting down")
                peer_gone.set()

    read_writer, read_stream = create_context_streams[SessionMessage | Exception](0)
    write_stream, write_reader = create_context_streams[SessionMessage](0)

    async def read_frame() -> bytes | None:
        """Read one line without ever buffering more than MAX_LINE_BYTES (+1).

        Returns the line, ``b""`` for an oversized frame (its remainder is discarded up to the
        next newline), or ``None`` at EOF."""
        line = await anyio.to_thread.run_sync(wire_in.readline, MAX_LINE_BYTES + 1)
        if not line:
            return None
        if len(line) <= MAX_LINE_BYTES or line.endswith(b"\n"):
            return line
        while True:  # oversized: drain in bounded chunks so memory stays flat
            rest = await anyio.to_thread.run_sync(wire_in.readline, 65536)
            if not rest or rest.endswith(b"\n"):
                return b""

    async def stdin_reader() -> None:
        try:
            async with read_writer:
                while (raw := await read_frame()) is not None:
                    if peer_gone.is_set():
                        break
                    if raw == b"":
                        logger.warning("dropping oversized frame (> %d bytes)", MAX_LINE_BYTES)
                        await emit(_error_frame(types.INVALID_REQUEST, "Request too large"))
                        continue
                    if not raw.strip():
                        continue
                    try:
                        payload = json.loads(raw.decode("utf-8"), parse_int=_bounded_int)
                    except _NumberTooLarge:
                        await emit(_error_frame(types.INVALID_REQUEST, "Number too large"))
                        continue
                    except (UnicodeDecodeError, ValueError):
                        logger.warning("parse error on inbound frame")
                        await emit(_error_frame(types.PARSE_ERROR, "Parse error"))
                        continue
                    if isinstance(payload, list):
                        await emit(_error_frame(types.INVALID_REQUEST, "Batch requests are not supported"))
                        continue
                    if _has_lone_surrogate(payload):
                        # "\ud800" is valid JSON but not valid Unicode: it would crash UTF-8 serialisation later.
                        if not _looks_like_notification(payload):
                            await emit(_error_frame(types.INVALID_REQUEST, "Invalid Request: unpaired surrogate"))
                        continue
                    try:
                        message = types.jsonrpc_message_adapter.validate_python(payload, by_name=False)
                    except ValidationError:
                        logger.warning("invalid JSON-RPC envelope")
                        if _looks_like_notification(payload):
                            continue
                        await emit(_error_frame(types.INVALID_REQUEST, "Invalid Request", _recover_id(payload)))
                        continue
                    if isinstance(message, types.JSONRPCNotification) and isinstance(payload, dict) and "id" in payload:
                        # e.g. "id": true / 1.0 / {} / null: the SDK would treat it as a notification and the
                        # caller would wait forever. MCP requires string/integer, non-null ids.
                        await emit(_error_frame(types.INVALID_REQUEST, "Invalid Request: id must be a string or integer"))
                        continue
                    await read_writer.send(SessionMessage(message))
        except anyio.ClosedResourceError:  # pragma: no cover - shutdown race
            await anyio.lowlevel.checkpoint()

    async def stdout_writer() -> None:
        try:
            async with write_reader:
                async for session_message in write_reader:
                    msg = session_message.message
                    try:
                        frame = msg.model_dump_json(by_alias=True, exclude_unset=True)
                    except Exception:
                        # Never let one unserialisable response kill the server or corrupt the wire.
                        logger.exception("failed to serialise outbound message")
                        rid = getattr(msg, "id", None)
                        frame = _error_frame(types.INTERNAL_ERROR, "Internal error", rid if _recover_id({"id": rid}) is not None else None)
                    await emit(frame)
        except anyio.ClosedResourceError:  # pragma: no cover - shutdown race
            await anyio.lowlevel.checkpoint()

    async with anyio.create_task_group() as tg:
        tg.start_soon(stdin_reader)
        tg.start_soon(stdout_writer)
        yield read_stream, write_stream
