"""Task 2: MCP security gateway tests against a *real* SDK-based downstream MCP server."""

from __future__ import annotations

import json

import httpx
import jwt
import pytest

from task2_mcp_gateway.auth import TokenVerifier
from task2_mcp_gateway.gateway import GatewaySettings, create_app
from task2_mcp_gateway.mock_downstream import build as build_downstream
from task2_mcp_gateway.policy import is_admin_tool

from .conftest import run_uvicorn

SECRET = "test-secret-that-is-definitely-32-chars-long"
ACCEPT = "application/json, text/event-stream"
UNAUTHORIZED = -32001


@pytest.fixture(scope="module", params=["json", "sse"])
def stack(request):
    """Downstream (JSON or SSE response mode) + gateway, each on its own real port."""
    downstream_app, calls = build_downstream(json_response=request.param == "json")
    with run_uvicorn(downstream_app) as downstream_url:
        settings = GatewaySettings(downstream_url=f"{downstream_url}/mcp", jwt_secret=SECRET)
        with run_uvicorn(create_app(settings)) as gateway_url:
            yield {"gateway": gateway_url, "calls": calls, "mode": request.param, "downstream": downstream_url}


@pytest.fixture
def tokens():
    v = TokenVerifier(SECRET)
    return {"admin": v.mint("alice", "admin"), "viewer": v.mint("bob", "viewer")}


def parse_body(resp: httpx.Response):
    if resp.headers.get("content-type", "").startswith("text/event-stream"):
        msgs = [json.loads(line[5:]) for line in resp.text.splitlines() if line.startswith("data:") and line[5:].strip()]
        return msgs[0] if len(msgs) == 1 else msgs
    return resp.json()


class McpHttpSession:
    """Tiny raw Streamable-HTTP client so we can see exact wire behaviour."""

    def __init__(self, base: str, token: str | None):
        self.client = httpx.Client(base_url=base, timeout=10)
        self.token = token
        self.session_id: str | None = None
        self._id = 0

    def headers(self, extra: dict | None = None) -> dict:
        h = {"accept": ACCEPT, "content-type": "application/json", "mcp-protocol-version": "2025-11-25"}
        if self.token:
            h["authorization"] = f"Bearer {self.token}"
        if self.session_id:
            h["mcp-session-id"] = self.session_id
        h.update(extra or {})
        return h

    def post(self, payload, headers: dict | None = None, raw: bytes | None = None) -> httpx.Response:
        return self.client.post("/mcp", content=raw if raw is not None else json.dumps(payload), headers=self.headers(headers))

    def rpc(self, method: str, params: dict | None = None, headers: dict | None = None) -> tuple[httpx.Response, dict]:
        self._id += 1
        msg = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            msg["params"] = params
        resp = self.post(msg, headers)
        return resp, parse_body(resp)

    def initialize(self) -> McpHttpSession:
        resp, body = self.rpc(
            "initialize", {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}
        )
        assert resp.status_code == 200, resp.text
        self.session_id = resp.headers.get("mcp-session-id")
        assert self.session_id, "gateway must pass Mcp-Session-Id back to the client"
        assert self.post({"jsonrpc": "2.0", "method": "notifications/initialized"}).status_code == 202
        return self


# ----------------------------------------------------------------------------- authentication
def test_missing_token_is_401(stack):
    resp, body = McpHttpSession(stack["gateway"], None).rpc("tools/list")
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"].startswith("Bearer")
    assert body["error"]["code"] == -32000


@pytest.mark.parametrize(
    "token",
    [
        "not-a-jwt",
        jwt.encode({"sub": "x", "role": "admin", "exp": 9999999999}, "wrong-secret-wrong-secret-wrong-secret!!", "HS256"),
        jwt.encode({"sub": "x", "role": "admin", "exp": 1}, SECRET, "HS256"),  # expired
        jwt.encode({"sub": "x", "role": "superuser", "exp": 9999999999}, SECRET, "HS256"),  # unknown role
        jwt.encode({"sub": "x", "exp": 9999999999}, SECRET, "HS256"),  # no role
        jwt.encode({"sub": "x", "role": "admin", "exp": 9999999999}, None, "none"),  # alg=none
    ],
    ids=["garbage", "bad-signature", "expired", "unknown-role", "no-role", "alg-none"],
)
def test_invalid_tokens_rejected(stack, token):
    resp, _ = McpHttpSession(stack["gateway"], token).rpc("tools/list")
    assert resp.status_code == 401


def test_non_bearer_scheme_rejected(stack, tokens):
    s = McpHttpSession(stack["gateway"], None)
    resp, _ = s.rpc("tools/list", headers={"authorization": f"Basic {tokens['admin']}"})
    assert resp.status_code == 401


# ----------------------------------------------------------------------------- forwarding
def test_tools_list_forwarded_transparently_for_viewer(stack, tokens):
    s = McpHttpSession(stack["gateway"], tokens["viewer"]).initialize()
    resp, body = s.rpc("tools/list")
    assert resp.status_code == 200
    names = {t["name"] for t in body["result"]["tools"]}
    assert {"admin_reset_key", "get_customer_record"} <= names  # transparent: nothing filtered by default


def test_viewer_can_call_regular_tool(stack, tokens):
    s = McpHttpSession(stack["gateway"], tokens["viewer"]).initialize()
    resp, body = s.rpc("tools/call", {"name": "get_customer_record", "arguments": {"customer_id": "CUST-10001"}})
    assert resp.status_code == 200
    assert "error" not in body
    assert stack["calls"][-1]["name"] == "get_customer_record"


# ----------------------------------------------------------------------------- authorization
def test_viewer_admin_tool_blocked_and_never_reaches_downstream(stack, tokens):
    s = McpHttpSession(stack["gateway"], tokens["viewer"]).initialize()
    before = len(stack["calls"])
    resp, body = s.rpc("tools/call", {"name": "admin_reset_key", "arguments": {"tenant": "acme"}})
    assert resp.status_code == 200
    assert resp.headers["x-gateway-decision"] == "deny"
    assert body == {
        "jsonrpc": "2.0",
        "id": s._id,
        "error": {
            "code": UNAUTHORIZED,
            "message": "Unauthorized Tool Call",
            "data": {"reason": "forbidden_tool", "tool": "admin_reset_key", "required_role": "admin"},
        },
    }
    assert len(stack["calls"]) == before, "denied call must not hit downstream"


def test_admin_can_call_admin_tool(stack, tokens):
    s = McpHttpSession(stack["gateway"], tokens["admin"]).initialize()
    resp, body = s.rpc("tools/call", {"name": "admin_reset_key", "arguments": {"tenant": "acme"}})
    assert resp.status_code == 200 and "result" in body
    assert stack["calls"][-1] == {"name": "admin_reset_key", "arguments": {"tenant": "acme"}}


@pytest.mark.parametrize(
    "name",
    ["ADMIN_reset_key", "Admin_Reset_Key", "ａｄｍｉｎ_reset_key", "admin​_reset_key", " admin_reset_key", "​admin_x"],
)
def test_lookalike_admin_names_blocked(stack, tokens, name):
    assert is_admin_tool(name)
    s = McpHttpSession(stack["gateway"], tokens["viewer"]).initialize()
    before = len(stack["calls"])
    _, body = s.rpc("tools/call", {"name": name, "arguments": {}})
    assert body["error"]["code"] == UNAUTHORIZED
    assert len(stack["calls"]) == before


def test_non_admin_names_containing_admin_are_allowed():
    assert not is_admin_tool("get_admin_contacts")
    assert not is_admin_tool("administer")  # no underscore -> not the admin_ namespace


def test_duplicate_key_smuggling_rejected(stack, tokens):
    s = McpHttpSession(stack["gateway"], tokens["viewer"]).initialize()
    raw = b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_customer_record","name":"admin_reset_key"}}'
    before = len(stack["calls"])
    resp = s.post(None, raw=raw)
    assert resp.status_code == 400 and resp.json()["error"]["code"] == -32600
    assert len(stack["calls"]) == before


@pytest.mark.parametrize(
    "headers",
    [
        {"mcp-name": "get_customer_record"},
        {"mcp-name": "=?base64?Z2V0X2N1c3RvbWVyX3JlY29yZA==?="},  # base64("get_customer_record")
        {"mcp-method": "tools/list"},
    ],
)
def test_mirror_header_mismatch_rejected(stack, tokens, headers):
    """2026-07-28 Mcp-Method/Mcp-Name headers must agree with the body (-32020 HeaderMismatch)."""
    s = McpHttpSession(stack["gateway"], tokens["admin"]).initialize()
    before = len(stack["calls"])
    resp, body = s.rpc("tools/call", {"name": "admin_reset_key", "arguments": {"tenant": "x"}}, headers=headers)
    assert resp.status_code == 400 and body["error"]["code"] == -32020
    assert len(stack["calls"]) == before


def test_mirror_header_base64_match_accepted(stack, tokens):
    s = McpHttpSession(stack["gateway"], tokens["viewer"]).initialize()
    resp, body = s.rpc(
        "tools/call",
        {"name": "list_open_tickets", "arguments": {}},
        headers={"mcp-method": "tools/call", "mcp-name": "=?base64?bGlzdF9vcGVuX3RpY2tldHM=?="},
    )
    assert resp.status_code == 200 and "result" in body


@pytest.mark.parametrize(
    "raw",
    [
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"list_open_tickets","Name":"admin_reset_key"}}',
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call","Method":"tools/list","params":{"name":"admin_reset_key"}}',
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"NAME":"admin_reset_key"}}',
    ],
)
def test_case_variant_key_smuggling_rejected(stack, tokens, raw):
    """Envoy AI Gateway / Go SDK CVE-2026-27896 style: case-insensitive decoders bind "Name" to name."""
    s = McpHttpSession(stack["gateway"], tokens["viewer"]).initialize()
    before = len(stack["calls"])
    resp = s.post(None, raw=raw)
    assert resp.status_code == 400 and resp.json()["error"]["code"] == -32600
    assert len(stack["calls"]) == before


def test_tools_call_notification_never_forwarded(stack, tokens):
    s = McpHttpSession(stack["gateway"], tokens["admin"]).initialize()
    before = len(stack["calls"])
    resp = s.post({"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "admin_reset_key", "arguments": {}}})
    assert resp.status_code == 400
    assert len(stack["calls"]) == before


def test_tool_name_outside_charset_rejected(stack, tokens):
    s = McpHttpSession(stack["gateway"], tokens["admin"]).initialize()
    _, body = s.rpc("tools/call", {"name": "list_open_tickets\u0000", "arguments": {}})
    assert body["error"]["code"] == -32602


def test_session_bound_to_subject(stack):
    v = TokenVerifier(SECRET)
    alice = McpHttpSession(stack["gateway"], v.mint("alice-2", "viewer")).initialize()
    mallory = McpHttpSession(stack["gateway"], v.mint("mallory", "viewer"))
    mallory.session_id = alice.session_id  # stolen / guessed session id
    resp, _ = mallory.rpc("tools/list")
    assert resp.status_code == 404
    assert alice.rpc("tools/list")[0].status_code == 200


def test_foreign_origin_rejected(stack, tokens):
    resp, body = McpHttpSession(stack["gateway"], tokens["admin"]).rpc("tools/list", headers={"origin": "https://evil.example"})
    assert resp.status_code == 403 and body["error"]["code"] == -32004


def test_multiple_authorization_headers_rejected(stack, tokens):
    resp = httpx.post(
        f"{stack['gateway']}/mcp",
        content=b'{"jsonrpc":"2.0","id":1,"method":"tools/list"}',
        headers=[("authorization", f"Bearer {tokens['viewer']}"), ("authorization", f"Bearer {tokens['admin']}"),
                 ("content-type", "application/json"), ("accept", ACCEPT)],
    )
    assert resp.status_code == 400


def test_client_token_not_forwarded_downstream(tokens):
    """Token passthrough is a MUST NOT in the MCP authorization spec."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = b'{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}'
        return httpx.Response(200, headers={"content-type": "application/json"}, stream=httpx.ByteStream(body))

    settings = GatewaySettings(
        downstream_url="http://downstream/mcp", jwt_secret=SECRET, downstream_headers={"Authorization": "Bearer svc"}
    )
    app = create_app(settings, transport=httpx.MockTransport(handler))
    with run_uvicorn(app) as url:
        McpHttpSession(url, tokens["admin"]).rpc("tools/list", headers={"cookie": "session=abc"})
    fwd = seen[0].headers
    assert fwd["authorization"] == "Bearer svc"
    assert "cookie" not in fwd
    assert fwd["x-gateway-subject"] == "alice" and fwd["x-gateway-role"] == "admin"


def test_batches_rejected_and_never_forwarded(stack, tokens):
    """MCP removed JSON-RPC batching in 2025-06-18; refusing arrays closes the batch-smuggling class entirely."""
    s = McpHttpSession(stack["gateway"], tokens["viewer"]).initialize()
    before = len(stack["calls"])
    batch = [
        {"jsonrpc": "2.0", "id": "a", "method": "tools/call", "params": {"name": "list_open_tickets", "arguments": {}}},
        {"jsonrpc": "2.0", "id": "b", "method": "tools/call", "params": {"name": "admin_delete_user", "arguments": {}}},
    ]
    resp = s.post(batch, headers={"mcp-method": "tools/list"})
    assert resp.status_code == 400 and resp.json()["error"]["code"] == -32600
    assert len(stack["calls"]) == before


# ----------------------------------------------------------------------------- malformed input
def test_parse_error(stack, tokens):
    resp = McpHttpSession(stack["gateway"], tokens["viewer"]).post(None, raw=b"{nope")
    assert resp.status_code == 400 and resp.json()["error"]["code"] == -32700


@pytest.mark.parametrize(
    "payload",
    [
        {"id": 1, "method": "tools/list"},  # missing jsonrpc
        {"jsonrpc": "1.0", "id": 1, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": {"x": 1}, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 1, "method": 42},
        "just a string",
    ],
)
def test_invalid_envelopes(stack, tokens, payload):
    resp = McpHttpSession(stack["gateway"], tokens["viewer"]).post(payload)
    assert resp.status_code == 400 and resp.json()["error"]["code"] == -32600


@pytest.mark.parametrize("params", [{}, {"name": 123}, {"name": ""}, None])
def test_tools_call_without_valid_name(stack, tokens, params):
    msg = {"jsonrpc": "2.0", "id": 9, "method": "tools/call"}
    if params is not None:
        msg["params"] = params
    resp = McpHttpSession(stack["gateway"], tokens["viewer"]).post(msg)
    assert resp.json()["error"]["code"] == -32602 and resp.json()["id"] == 9


def test_body_size_limit(stack, tokens):
    resp = McpHttpSession(stack["gateway"], tokens["viewer"]).post(None, raw=b" " * 1_100_000)
    assert resp.status_code == 413


def test_downstream_unavailable_is_sanitised(tokens):
    app = create_app(GatewaySettings(downstream_url="http://127.0.0.1:1/mcp", jwt_secret=SECRET, connect_timeout=1))
    with run_uvicorn(app) as url:
        resp, body = McpHttpSession(url, tokens["viewer"]).rpc("tools/list")
    assert resp.status_code == 502
    assert body["error"] == {"code": -32002, "message": "Downstream MCP server unavailable"}


# ----------------------------------------------------------------------------- optional discovery filter
def test_optional_tools_list_filter(tokens):
    for json_mode in (True, False):
        downstream_app, _ = build_downstream(json_response=json_mode)
        with run_uvicorn(downstream_app) as d:
            app = create_app(GatewaySettings(downstream_url=f"{d}/mcp", jwt_secret=SECRET, filter_tools_list=True))
            with run_uvicorn(app) as g:
                viewer = McpHttpSession(g, tokens["viewer"]).initialize()
                admin = McpHttpSession(g, tokens["admin"]).initialize()
                viewer_names = {t["name"] for t in viewer.rpc("tools/list")[1]["result"]["tools"]}
                admin_names = {t["name"] for t in admin.rpc("tools/list")[1]["result"]["tools"]}
        assert not any(n.startswith("admin_") for n in viewer_names)
        assert {"admin_reset_key", "admin_delete_user"} <= admin_names


# ----------------------------------------------------------------------------- end-to-end with the official SDK client
async def test_official_sdk_client_through_gateway(stack, tokens):
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared.exceptions import MCPError

    async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {tokens['viewer']}"}, timeout=10) as http:
        async with streamable_http_client(f"{stack['gateway']}/mcp", http_client=http) as (read, write, *_):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert "admin_reset_key" in {t.name for t in tools.tools}
                ok = await session.call_tool("list_open_tickets", {})
                assert not ok.is_error
                with pytest.raises(MCPError) as exc:
                    await session.call_tool("admin_reset_key", {"tenant": "acme"})
                assert exc.value.code == UNAUTHORIZED
                assert exc.value.message == "Unauthorized Tool Call"


# ----------------------------------------------------------------------------- iteration 3 (adversarial review regressions)
def _raw_post(url: str, headers: list[tuple[str, str]], body: bytes, method: str = "POST") -> httpx.Response:
    base = [("content-type", "application/json"), ("accept", ACCEPT), ("mcp-protocol-version", "2025-11-25")]
    return httpx.request(method, f"{url}/mcp", content=body, headers=base + headers, timeout=10)


def test_duplicate_session_header_cannot_bypass_binding(stack):
    v = TokenVerifier(SECRET)
    alice = McpHttpSession(stack["gateway"], v.mint("alice-dup", "viewer")).initialize()
    bob = v.mint("bob-dup", "viewer")
    body = b'{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
    for method in ("POST", "DELETE"):
        resp = _raw_post(stack["gateway"], [("authorization", f"Bearer {bob}"), ("mcp-session-id", "decoy"),
                                             ("mcp-session-id", alice.session_id)], body if method == "POST" else b"", method)
        assert resp.status_code == 400, method
    assert alice.rpc("tools/list")[0].status_code == 200  # alice's session untouched


def test_unsigned_or_raw_downstream_session_id_rejected(stack, tokens):
    s = McpHttpSession(stack["gateway"], tokens["viewer"]).initialize()
    raw_downstream_id = s.session_id.rpartition(".")[0]
    for forged in (raw_downstream_id, raw_downstream_id + ".AAAAAAAAAAAAAAAAAAAAAA", "x.y"):
        s2 = McpHttpSession(stack["gateway"], tokens["viewer"])
        s2.session_id = forged
        assert s2.rpc("tools/list")[0].status_code == 404, forged


def test_session_binding_survives_gateway_restart(stack, tokens):
    """HMAC binding is stateless: a new gateway instance with the same secret still enforces it."""
    v = TokenVerifier(SECRET)
    alice = McpHttpSession(stack["gateway"], v.mint("alice-r", "viewer")).initialize()
    # A second, freshly started gateway (same secret, no shared state) honours alice's id and rejects mallory.
    fresh = create_app(GatewaySettings(downstream_url=stack["downstream"] + "/mcp", jwt_secret=SECRET))
    with run_uvicorn(fresh) as fresh_url:
        alice2 = McpHttpSession(fresh_url, alice.token)
        alice2.session_id = alice.session_id
        assert alice2.rpc("tools/list")[0].status_code == 200
        intruder = McpHttpSession(fresh_url, v.mint("mallory-r", "viewer"))
        intruder.session_id = alice.session_id
        assert intruder.rpc("tools/list")[0].status_code == 404
    mallory = McpHttpSession(stack["gateway"], v.mint("mallory-r", "viewer"))
    mallory.session_id = alice.session_id
    assert mallory.rpc("tools/list")[0].status_code == 404
    assert alice.rpc("tools/list")[0].status_code == 200


@pytest.mark.parametrize(
    "raw",
    [
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_customer_record","arguments":{"customer_id":"\\ud800"}}}',
        b'{"jsonrpc":"2.0","id":"\\udfff","method":"tools/list"}',
        b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"n":' + b"9" * 5000 + b"}}",
    ],
    ids=["surrogate-arg", "surrogate-id", "huge-int"],
)
def test_unsafe_json_is_invalid_request_not_500(stack, tokens, raw):
    s = McpHttpSession(stack["gateway"], tokens["viewer"]).initialize()
    resp = s.post(None, raw=raw)
    assert resp.status_code == 400 and resp.json()["error"]["code"] == -32600


def test_null_request_id_rejected(stack, tokens):
    resp = McpHttpSession(stack["gateway"], tokens["admin"]).post(
        {"jsonrpc": "2.0", "id": None, "method": "tools/call", "params": {"name": "admin_reset_key", "arguments": {}}}
    )
    assert resp.status_code == 400 and resp.json()["error"]["code"] == -32600


def test_client_cannot_choose_request_id(stack, tokens):
    resp, _ = McpHttpSession(stack["gateway"], tokens["viewer"]).rpc("tools/list", headers={"x-request-id": "spoofed"})
    assert resp.headers["x-request-id"] != "spoofed"


def _mock_gateway(handler, **settings_kw):
    settings = GatewaySettings(downstream_url="http://downstream/mcp", jwt_secret=SECRET, **settings_kw)
    return create_app(settings, transport=httpx.MockTransport(handler))


def test_response_headers_preserved_correctly(tokens):
    """Content-Encoding kept with raw bytes; repeated Set-Cookie not merged; Connection-named headers dropped."""
    import gzip

    payload = gzip.compress(b'{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}')

    def handler(request: httpx.Request) -> httpx.Response:
        headers = [("content-type", "application/json"), ("content-encoding", "gzip"), ("set-cookie", "a=1"),
                   ("set-cookie", "b=2"), ("connection", "x-internal"), ("x-internal", "secret-topology")]
        return httpx.Response(200, headers=headers, stream=httpx.ByteStream(payload))

    with run_uvicorn(_mock_gateway(handler)) as url:
        resp = McpHttpSession(url, tokens["viewer"]).post({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert resp.json() == {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}  # httpx decoded the gzip
    assert resp.headers.get_list("set-cookie") == ["a=1", "b=2"]
    assert "x-internal" not in resp.headers


def test_filtered_tools_list_handles_gzip(tokens):
    import gzip

    body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"tools": [{"name": "admin_x"}, {"name": "ok"}]}}).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json", "content-encoding": "gzip"},
                              stream=httpx.ByteStream(gzip.compress(body)))

    with run_uvicorn(_mock_gateway(handler, filter_tools_list=True)) as url:
        resp = McpHttpSession(url, tokens["viewer"]).post({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert [t["name"] for t in resp.json()["result"]["tools"]] == ["ok"]


def test_get_streams_capped_per_subject(tokens):
    import asyncio as _asyncio

    async def handler(request: httpx.Request) -> httpx.Response:
        async def forever():
            yield b": keep-alive\n\n"
            await _asyncio.sleep(30)
            yield b""
        if request.method == "GET":
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_AsyncStream(forever()))
        return httpx.Response(200, headers={"content-type": "application/json"},
                              stream=httpx.ByteStream(b'{"jsonrpc":"2.0","id":1,"result":{}}'))

    with run_uvicorn(_mock_gateway(handler, max_streams_per_subject=2)) as url:
        h = {"authorization": f"Bearer {tokens['viewer']}", "accept": "text/event-stream"}
        with httpx.Client(timeout=10) as c:
            streams = [c.stream("GET", f"{url}/mcp", headers=h) for _ in range(2)]
            opened = [s.__enter__() for s in streams]
            assert all(r.status_code == 200 for r in opened)
            third = c.get(f"{url}/mcp", headers=h)
            assert third.status_code == 429
            # another subject is unaffected, and POSTs still work for everyone
            admin_h = {"authorization": f"Bearer {tokens['admin']}", "accept": "text/event-stream"}
            with c.stream("GET", f"{url}/mcp", headers=admin_h) as r:
                assert r.status_code == 200
            assert McpHttpSession(url, tokens["viewer"]).rpc("ping")[0].status_code == 200
            for s in streams:
                s.__exit__(None, None, None)


class _AsyncStream(httpx.AsyncByteStream):
    def __init__(self, gen):
        self._gen = gen

    async def __aiter__(self):
        async for chunk in self._gen:
            yield chunk
