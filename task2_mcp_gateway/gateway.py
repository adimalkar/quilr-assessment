"""Task 2 - MCP Security Gateway: authenticating, tool-authorizing JSON-RPC reverse proxy.

    agent ──HTTP(S)──► gateway /mcp ──► downstream MCP server (Streamable HTTP)
                         │
                         ├─ 1. authenticate   Authorization: Bearer <JWT{sub, role}>
                         ├─ 2. parse          strict JSON-RPC 2.0 (no dup keys, no NaN)
                         ├─ 3. authorize      tools/call + admin_* name ⇒ role must be admin
                         │                    else -32001 "Unauthorized Tool Call", downstream never hit
                         └─ 4. forward        streamed byte-for-byte (JSON or SSE); Mcp-Session-Id is
                                              HMAC-bound to the caller's JWT subject

Run:  GATEWAY_JWT_SECRET=... GATEWAY_DOWNSTREAM_URL=http://127.0.0.1:9001/mcp \
      uv run uvicorn --factory task2_mcp_gateway.gateway:app_factory --port 9000
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from task3_llm_guardrail.sse import SSEDecoder, encode_event

from . import jsonrpc as rpc
from .auth import AuthError, Principal, TokenVerifier
from .policy import authorize_tool_call, tool_visible
from .sessions import SessionBinder

logger = logging.getLogger("task2.gateway")
audit_log = logging.getLogger("task2.gateway.audit")

# Request headers we pass downstream. Everything else - notably the caller's Authorization,
# Cookie, and hop-by-hop headers - is dropped. Mcp-* covers session id, protocol version and
# the 2026-07-28 Mcp-Method / Mcp-Name / Mcp-Param-* mirror headers.
_FORWARD_REQUEST_HEADERS = {"content-type", "accept", "last-event-id", "user-agent"}
# RFC 9110 §7.6.1 hop-by-hop headers (+ content-length, which the ASGI server recomputes).
# Content-Encoding is end-to-end and is preserved on the streaming path.
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection", "te",
    "trailer", "transfer-encoding", "upgrade", "content-length",
}


@dataclass
class GatewaySettings:
    downstream_url: str
    jwt_secret: str
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    # Spec says tools/list is forwarded transparently; enable to also hide admin_* tools from
    # non-admins (least-privilege discovery). Enforcement on tools/call happens regardless.
    filter_tools_list: bool = False
    max_body_bytes: int = 1_000_000
    connect_timeout: float = 5.0
    # POST responses (JSON or SSE) may legitimately stream for a while (progress, long tools)
    read_timeout: float | None = 300.0
    downstream_headers: dict[str, str] = field(default_factory=dict)
    # Browser Origins allowed to talk to the gateway (MCP transport spec: servers MUST validate Origin
    # to stop DNS-rebinding). Requests without an Origin header (agents, CLIs) are unaffected.
    allowed_origins: frozenset[str] = frozenset()
    # Long-lived GET (server->client SSE) streams use their own connection pool and are capped per
    # subject, so one user can't exhaust the connections every other user's POSTs need.
    max_streams_per_subject: int = 4
    max_streams_total: int = 500

    @classmethod
    def from_env(cls) -> GatewaySettings:
        extra = {}
        if token := os.environ.get("GATEWAY_DOWNSTREAM_TOKEN"):
            extra["Authorization"] = f"Bearer {token}"
        return cls(
            downstream_url=os.environ.get("GATEWAY_DOWNSTREAM_URL", "http://127.0.0.1:9001/mcp"),
            jwt_secret=os.environ["GATEWAY_JWT_SECRET"],
            jwt_issuer=os.environ.get("GATEWAY_JWT_ISSUER"),
            jwt_audience=os.environ.get("GATEWAY_JWT_AUDIENCE"),
            filter_tools_list=os.environ.get("GATEWAY_FILTER_TOOLS_LIST", "0").lower() in {"1", "true", "yes"},
            allowed_origins=frozenset(o for o in os.environ.get("GATEWAY_ALLOWED_ORIGINS", "").split(",") if o),
            downstream_headers=extra,
        )


class _Reject(Exception):
    """Short-circuit with a locally generated JSON-RPC response."""

    def __init__(self, status: int, body: dict[str, Any] | None, headers: dict[str, str] | None = None):
        self.status, self.body, self.headers = status, body, headers or {}


def _json(status: int, body: Any, headers: dict[str, str] | None = None) -> Response:
    return Response(rpc.dumps(body), status_code=status, media_type="application/json", headers=headers)


def create_app(settings: GatewaySettings, *, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    verifier = TokenVerifier(settings.jwt_secret, issuer=settings.jwt_issuer, audience=settings.jwt_audience)
    sessions = SessionBinder(settings.jwt_secret)
    client = httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(connect=settings.connect_timeout, read=settings.read_timeout, write=30.0, pool=5.0),
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        follow_redirects=False,  # never let downstream bounce us (and our headers) elsewhere
    )
    stream_client = httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(connect=settings.connect_timeout, read=None, write=30.0, pool=5.0),
        limits=httpx.Limits(max_connections=settings.max_streams_total, max_keepalive_connections=0),
        follow_redirects=False,
    )
    open_streams: Counter[str] = Counter()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await client.aclose()
        await stream_client.aclose()

    app = FastAPI(title="MCP Security Gateway", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings
    app.state.verifier = verifier
    app.state.open_streams = open_streams

    # ------------------------------------------------------------------ helpers
    def audit(ctx: dict[str, Any], **fields: Any) -> None:
        audit_log.info(json.dumps({"ts": round(time.time(), 3), **ctx, **fields}, default=str))

    def new_rid(request: Request) -> tuple[str, dict[str, str]]:
        """Gateway-generated id (client-supplied X-Request-Id can't be used to spoof audit correlation)."""
        rid = uuid.uuid4().hex
        client_rid = request.headers.get("x-request-id")
        return rid, ({"client_rid": client_rid[:128]} if client_rid else {})

    def check_origin(request: Request) -> None:
        origin = request.headers.get("origin")
        if origin is not None and origin not in settings.allowed_origins:
            raise _Reject(403, rpc.error(None, rpc.FORBIDDEN, "Origin not allowed"))

    def check_single_valued_headers(request: Request) -> None:
        """Repeated security-relevant headers are ambiguous (first-wins vs last-wins vs comma-join):
        the value we check must be the value we forward, so refuse ambiguity outright."""
        seen = Counter(k for k, _ in request.headers.items())
        for name, count in seen.items():
            if count > 1 and (name in _FORWARD_REQUEST_HEADERS or name.startswith("mcp-") or name == "authorization"):
                raise _Reject(400, rpc.error(None, rpc.INVALID_REQUEST, f"Header {name!r} must not be repeated"))

    def authenticate(request: Request) -> tuple[Principal, str | None]:
        """Returns (principal, downstream session id or None)."""
        check_origin(request)
        check_single_valued_headers(request)
        try:
            principal = verifier.verify_header(request.headers.get("authorization"))
        except AuthError as exc:
            raise _Reject(
                401,
                rpc.error(None, rpc.AUTHENTICATION_REQUIRED, "Authentication required", {"reason": exc.message}),
                {"WWW-Authenticate": 'Bearer error="invalid_token"'},
            ) from None
        client_sid = request.headers.get("mcp-session-id")
        if client_sid is None:
            return principal, None
        downstream_sid = sessions.unwrap(principal.subject, client_sid)
        if downstream_sid is None:
            logger.warning("rejected session id not issued to subject %s", principal.subject)
            # 404 is the spec's "unknown session" answer: it doesn't reveal whether it exists for someone else.
            raise _Reject(404, rpc.error(None, rpc.INVALID_REQUEST, "Session not found"))
        return principal, downstream_sid

    def downstream_headers(request: Request, principal: Principal, rid: str, sid: str | None) -> dict[str, str]:
        headers = {
            k: v for k, v in request.headers.items()
            if (k in _FORWARD_REQUEST_HEADERS or k.startswith("mcp-")) and k != "mcp-session-id"
        }
        if sid is not None:
            headers["mcp-session-id"] = sid  # exactly the value that was verified
        headers["accept-encoding"] = "identity"
        headers["x-forwarded-for"] = request.client.host if request.client else "unknown"
        headers["x-request-id"] = rid
        headers["x-gateway-subject"] = principal.subject
        headers["x-gateway-role"] = principal.role.value
        headers.update(settings.downstream_headers)
        return headers

    def copy_response_headers(response: Response, upstream: httpx.Response, principal: Principal, rid: str,
                              *, decoded: bool = False) -> Response:
        connection_named = {
            t.strip().lower() for v in upstream.headers.get_list("connection") for t in v.split(",") if t.strip()
        }
        drop = _HOP_BY_HOP | connection_named | ({"content-encoding"} if decoded else set())
        for k, v in upstream.headers.multi_items():  # multi_items: repeated headers (Set-Cookie) stay separate
            k = k.lower()
            if k in drop or k in response.headers and k == "content-type":
                continue
            if k == "mcp-session-id":
                v = sessions.wrap(principal.subject, v)
            response.headers.append(k, v)
        response.headers["x-request-id"] = rid
        if upstream.headers.get("content-type", "").startswith("text/event-stream"):
            response.headers["cache-control"] = "no-cache"
            response.headers["x-accel-buffering"] = "no"  # stop nginx & friends from buffering SSE
        return response

    async def read_body(request: Request) -> bytes:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.max_body_bytes:
            raise _Reject(413, rpc.error(None, rpc.INVALID_REQUEST, "Request body too large"))
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > settings.max_body_bytes:
                raise _Reject(413, rpc.error(None, rpc.INVALID_REQUEST, "Request body too large"))
            chunks.append(chunk)
        return b"".join(chunks)

    def check_mirror_headers(request: Request, msg: dict[str, Any]) -> None:
        """MCP 2026-07-28 mirrors method/name into headers. If present they MUST agree with the body,
        otherwise header-routing infrastructure and body-inspecting policy would see different calls."""
        method_hdr = request.headers.get("mcp-method")
        name_hdr = request.headers.get("mcp-name")
        if method_hdr is not None and method_hdr != msg.get("method"):
            raise _Reject(400, rpc.error(msg.get("id"), rpc.HEADER_MISMATCH, "Mcp-Method header does not match body"))
        if name_hdr is not None and msg.get("method") == "tools/call":
            params = msg.get("params")
            if not isinstance(params, dict) or rpc.decode_header_value(name_hdr) != params.get("name"):
                raise _Reject(400, rpc.error(msg.get("id"), rpc.HEADER_MISMATCH, "Mcp-Name header does not match body"))

    def policy_check(principal: Principal, msg: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any] | None:
        """Return a local JSON-RPC error response if the message is denied, else None."""
        if msg.get("method") != "tools/call":
            return None
        if "id" not in msg:
            # A tools/call notification is meaningless and would execute without any response to deny.
            audit(ctx, method="tools/call", decision="reject", reason="tools/call without id")
            raise _Reject(400, rpc.error(None, rpc.INVALID_REQUEST, "tools/call must be a request (missing id)"))
        params = msg.get("params")
        name = params.get("name") if isinstance(params, dict) else None
        if not isinstance(name, str) or not name:
            audit(ctx, method="tools/call", decision="reject", reason="missing tool name")
            return rpc.error(msg.get("id"), rpc.INVALID_PARAMS, "params.name must be a non-empty string")
        decision = authorize_tool_call(principal, name)
        audit(ctx, method="tools/call", tool=name, decision="allow" if decision.allowed else "deny", reason=decision.reason)
        if decision.allowed:
            return None
        if decision.code == "invalid_tool_name":
            return rpc.error(msg.get("id"), rpc.INVALID_PARAMS, "Invalid tool name", {"reason": decision.code})
        return rpc.error(
            msg.get("id"),
            rpc.UNAUTHORIZED_TOOL_CALL,
            "Unauthorized Tool Call",
            {"reason": decision.code, "tool": name, "required_role": "admin"},
        )

    async def send_downstream(method: str, request: Request, principal: Principal, rid: str, sid: str | None,
                              content: bytes | None, *, long_lived: bool = False) -> httpx.Response:
        http = stream_client if long_lived else client
        req = http.build_request(
            method, settings.downstream_url, content=content,
            headers=downstream_headers(request, principal, rid, sid), params=request.query_params,
        )
        try:
            return await http.send(req, stream=True)
        except httpx.TimeoutException:
            logger.warning("downstream timeout rid=%s", rid)
            raise _Reject(504, rpc.error(None, rpc.UPSTREAM_TIMEOUT, "Downstream MCP server timed out")) from None
        except httpx.HTTPError as exc:
            logger.warning("downstream unavailable rid=%s: %s", rid, type(exc).__name__)
            raise _Reject(502, rpc.error(None, rpc.UPSTREAM_UNAVAILABLE, "Downstream MCP server unavailable")) from None

    def stream_back(upstream: httpx.Response, principal: Principal, rid: str, on_close=None) -> StreamingResponse:
        async def body():
            # try/finally: on client disconnect Starlette cancels this generator, and we must release
            # the downstream connection (in 2026-07-28 a dropped connection *is* the cancellation signal).
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
                if on_close:
                    on_close()

        response = StreamingResponse(body(), status_code=upstream.status_code,
                                     media_type=upstream.headers.get("content-type"))
        return copy_response_headers(response, upstream, principal, rid)  # type: ignore[return-value]

    async def buffered(upstream: httpx.Response, limit: int = 10_000_000) -> bytes:
        try:
            data = bytearray()
            async for chunk in upstream.aiter_bytes():  # decoded (gzip etc.), since we will parse it
                data += chunk
                if len(data) > limit:
                    raise _Reject(502, rpc.error(None, rpc.UPSTREAM_UNAVAILABLE, "Downstream response too large"))
            return bytes(data)
        finally:
            await upstream.aclose()

    def filter_tools_payload(msg: Any, principal: Principal) -> Any:
        if isinstance(msg, dict) and isinstance(msg.get("result"), dict) and isinstance(msg["result"].get("tools"), list):
            msg["result"]["tools"] = [
                t for t in msg["result"]["tools"]
                if isinstance(t, dict) and isinstance(t.get("name"), str) and tool_visible(principal, t["name"])
            ]
        return msg

    def rewrite_body(body: bytes, content_type: str, transform) -> bytes:
        """Apply ``transform`` to each JSON-RPC message in a JSON or SSE body. Fails closed: anything
        that can't be parsed is dropped rather than passed through unfiltered."""
        if content_type.startswith("application/json"):
            return rpc.dumps(transform(json.loads(body)))
        if content_type.startswith("text/event-stream"):
            decoder, out = SSEDecoder(), []
            for ev in [*decoder.feed(body), *decoder.close()]:
                try:
                    out.append(encode_event(json.dumps(transform(json.loads(ev.data)), separators=(",", ":")), ev.event))
                except ValueError:
                    logger.warning("dropping unparseable SSE event from filtered tools/list")
            return b"".join(out)
        raise ValueError(f"cannot filter content type {content_type!r}")

    # ------------------------------------------------------------------ routes
    @app.exception_handler(_Reject)
    async def _on_reject(_: Request, exc: _Reject) -> Response:
        if exc.body is None:
            return Response(status_code=exc.status, headers=exc.headers)
        return _json(exc.status, exc.body, exc.headers)

    @app.exception_handler(Exception)
    async def _on_unexpected(_: Request, exc: Exception) -> Response:
        logger.exception("unhandled gateway error")
        return _json(500, rpc.error(None, rpc.INTERNAL_ERROR, "Internal gateway error"))

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/mcp")
    async def mcp_post(request: Request) -> Response:
        rid, extra = new_rid(request)
        principal, sid = authenticate(request)
        ctx = {"rid": rid, "sub": principal.subject, "role": principal.role.value, **extra}
        body = await read_body(request)

        try:
            msg = rpc.loads_strict(body)
        except rpc.UnsafeJSONError as exc:
            raise _Reject(400, rpc.error(None, rpc.INVALID_REQUEST, str(exc))) from None
        except (ValueError, UnicodeDecodeError):
            raise _Reject(400, rpc.error(None, rpc.PARSE_ERROR, "Parse error")) from None

        if isinstance(msg, list):
            # Batching was removed from MCP in 2025-06-18 and SDK servers reject arrays; refusing them also
            # removes a whole class of per-item policy/merge bugs. Nothing is forwarded.
            audit(ctx, method="<batch>", decision="reject", reason="batch")
            raise _Reject(400, rpc.error(None, rpc.INVALID_REQUEST, "Batch requests are not supported"))

        if problem := rpc.envelope_problem(msg) or rpc.smuggling_problem(msg):
            raise _Reject(400, rpc.error(msg.get("id") if isinstance(msg, dict) else None, rpc.INVALID_REQUEST, problem))
        check_mirror_headers(request, msg)

        if denied := policy_check(principal, msg, ctx):
            return _json(200, denied, {"x-request-id": rid, "x-gateway-decision": "deny"})

        method = msg.get("method")
        if method not in (None, "tools/call"):
            audit(ctx, method=method, decision="forward")

        # Forward the exact message we authorised (re-serialised), not the raw bytes.
        upstream = await send_downstream("POST", request, principal, rid, sid, rpc.dumps(msg))

        if method == "tools/list" and settings.filter_tools_list and not principal.is_admin and upstream.is_success:
            ctype = upstream.headers.get("content-type", "")
            data = await buffered(upstream)
            try:
                data = rewrite_body(data, ctype, lambda m: filter_tools_payload(m, principal))
            except (ValueError, UnicodeDecodeError):
                raise _Reject(502, rpc.error(msg.get("id"), rpc.UPSTREAM_UNAVAILABLE, "Unreadable downstream response")) from None
            response = Response(data, status_code=upstream.status_code, media_type=ctype or None)
            return copy_response_headers(response, upstream, principal, rid, decoded=True)

        return stream_back(upstream, principal, rid)

    @app.get("/mcp")
    async def mcp_get(request: Request) -> Response:
        """Server->client SSE stream (notifications). Authenticated, capped per subject, streamed through."""
        rid, _ = new_rid(request)
        principal, sid = authenticate(request)
        subject = principal.subject
        if open_streams[subject] >= settings.max_streams_per_subject:
            raise _Reject(429, rpc.error(None, rpc.FORBIDDEN, "Too many concurrent streams"), {"retry-after": "5"})
        open_streams[subject] += 1

        def release() -> None:
            open_streams[subject] -= 1
            if open_streams[subject] <= 0:
                del open_streams[subject]

        try:
            upstream = await send_downstream("GET", request, principal, rid, sid, None, long_lived=True)
        except BaseException:
            release()
            raise
        return stream_back(upstream, principal, rid, on_close=release)

    @app.delete("/mcp")
    async def mcp_delete(request: Request) -> Response:
        """Session termination (only the session's owner can get here - see authenticate())."""
        rid, _ = new_rid(request)
        principal, sid = authenticate(request)
        upstream = await send_downstream("DELETE", request, principal, rid, sid, None)
        return stream_back(upstream, principal, rid)

    return app


def app_factory() -> FastAPI:
    """uvicorn --factory entrypoint: settings come from the environment."""
    logging.basicConfig(
        level=os.environ.get("GATEWAY_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return create_app(GatewaySettings.from_env())
