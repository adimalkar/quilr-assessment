"""Task 1: black-box tests against the real server process over stdio.

Every test drives a subprocess through raw newline-delimited JSON-RPC and asserts
that *every* byte the server wrote to stdout is a valid JSON-RPC 2.0 frame.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from task1_mcp_server.schemas import GetCustomerRecordInput, TriggerRefundInput

ROOT = Path(__file__).resolve().parents[1]
INVALID_PARAMS, INVALID_REQUEST, PARSE_ERROR, METHOD_NOT_FOUND = -32602, -32600, -32700, -32601


class StdioHarness:
    def __init__(self, extra_code: str = "", module: str = "task1_mcp_server") -> None:
        # Run the real entry point (`python -m <module>`) after any injected test code.
        bootstrap = textwrap.dedent(extra_code) + f"\nimport runpy\nrunpy.run_module({module!r}, run_name='__main__')\n"
        self.proc = subprocess.Popen(
            [sys.executable, "-c", bootstrap],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        self.raw_stdout: list[bytes] = []
        self._lines: queue.Queue[bytes] = queue.Queue()
        self._stderr: list[bytes] = []
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=lambda: self._stderr.extend(self.proc.stderr), daemon=True).start()
        self._next_id = 100

    def _pump_stdout(self) -> None:
        for line in self.proc.stdout:
            self.raw_stdout.append(line)
            self._lines.put(line)

    def send_raw(self, text: str) -> None:
        self.proc.stdin.write(text.encode() + b"\n")
        self.proc.stdin.flush()

    def send(self, payload: dict) -> None:
        self.send_raw(json.dumps(payload))

    def recv(self, timeout: float = 5.0) -> dict:
        return json.loads(self._lines.get(timeout=timeout))

    def request(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        msg = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)
        resp = self.recv()
        assert resp["id"] == self._next_id
        return resp

    def initialize(self) -> dict:
        resp = self.request(
            "initialize",
            {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "1"}},
        )
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return resp

    def initialize_raw_tolerant(self) -> None:
        """Handshake that skips non-JSON stdout lines (used only by the negative-control test)."""
        self.send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "t", "version": "1"},
                },
            }
        )
        while True:
            line = self._lines.get(timeout=5)
            if line.startswith(b"{"):
                return

    def call(self, name: str, arguments: dict | None) -> dict:
        params: dict = {"name": name}
        if arguments is not None:
            params["arguments"] = arguments
        return self.request("tools/call", params)

    def close(self) -> str:
        self.proc.stdin.close()
        self.proc.wait(timeout=10)
        return b"".join(self._stderr).decode(errors="replace")

    def assert_stdout_pure(self) -> None:
        assert self.raw_stdout, "server wrote nothing to stdout"
        for raw in self.raw_stdout:
            assert raw.endswith(b"\n")
            frame = json.loads(raw)
            assert frame["jsonrpc"] == "2.0"
            assert ("result" in frame) ^ ("error" in frame) or "method" in frame


@pytest.fixture
def server():
    h = StdioHarness()
    h.initialize()
    yield h
    h.close()
    h.assert_stdout_pure()


# --------------------------------------------------------------------------- happy paths
def test_tools_list_schema_generated_from_pydantic(server):
    tools = {t["name"]: t for t in server.request("tools/list")["result"]["tools"]}
    assert set(tools) == {"get_customer_record", "trigger_refund"}
    assert tools["get_customer_record"]["inputSchema"] == {
        **{k: v for k, v in GetCustomerRecordInput.model_json_schema().items() if k != "title"}
    }
    refund = tools["trigger_refund"]["inputSchema"]
    assert refund["additionalProperties"] is False
    assert refund["properties"]["amount"]["exclusiveMinimum"] == 0
    assert refund["properties"]["reason"]["minLength"] == 10
    assert set(refund["required"]) == {"customer_id", "amount", "reason"}  # idempotency_key is optional
    assert set(refund["properties"]) == set(TriggerRefundInput.model_fields)


def test_get_customer_record_ok(server):
    result = server.call("get_customer_record", {"customer_id": "CUST-10001"})["result"]
    assert result["isError"] is False
    assert result["structuredContent"]["name"] == "Ada Lovelace"
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]


def test_trigger_refund_ok_and_balance_tracked(server):
    args = {"customer_id": "CUST-10002", "amount": 300.5, "reason": "Duplicate charge on invoice #42"}
    first = server.call("trigger_refund", args)["result"]
    assert first["isError"] is False
    assert first["structuredContent"]["remaining_refundable_balance"] == 10.0
    # Second refund exceeds remaining balance -> tool *execution* error (isError), not a protocol error
    second = server.call("trigger_refund", args)["result"]
    assert second["isError"] is True
    assert "exceeds refundable balance" in second["content"][0]["text"]


def test_unknown_customer_is_tool_execution_error(server):
    result = server.call("get_customer_record", {"customer_id": "CUST-99999"})["result"]
    assert result["isError"] is True


# --------------------------------------------------------------------------- validation -> -32602
@pytest.mark.parametrize(
    "customer_id",
    [
        "CUST-1234",
        "CUST-123456",
        "cust-12345",
        "CUST_12345",
        " CUST-12345",
        "CUST-12345\n",
        "CUST-1234a",
        "CUST-١٢٣٤٥",  # Arabic-Indic digits must not satisfy [0-9]
        "",
        12345,
        None,
        ["CUST-12345"],
    ],
)
def test_customer_id_format_rejected(server, customer_id):
    resp = server.call("get_customer_record", {"customer_id": customer_id})
    assert resp["error"]["code"] == INVALID_PARAMS
    assert resp["error"]["data"]["errors"][0]["field"] == "customer_id"


GOOD_REFUND = {"customer_id": "CUST-10003", "amount": 10.0, "reason": "Customer charged twice"}


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"amount": 0}, "amount"),
        ({"amount": -1.5}, "amount"),
        ({"amount": "10.00"}, "amount"),  # strict: no string coercion
        ({"amount": True}, "amount"),  # bool is not a number
        ({"amount": 10.001}, "amount"),  # sub-cent precision
        ({"amount": 1e12}, "amount"),  # above max
        ({"amount": None}, "amount"),
        ({"reason": "too short"}, "reason"),  # 9 chars
        ({"reason": "   short   "}, "reason"),  # whitespace stripped before length check
        ({"reason": "x" * 501}, "reason"),
        ({"reason": 1234567890}, "reason"),
        ({"customer_id": "CUST-1"}, "customer_id"),
        ({"unexpected": "field"}, "unexpected"),  # extra=forbid
    ],
)
def test_refund_validation_rejected(server, override, field):
    resp = server.call("trigger_refund", {**GOOD_REFUND, **override})
    err = resp["error"]
    assert err["code"] == INVALID_PARAMS
    assert field in {e["field"] for e in err["data"]["errors"]}


def test_nan_and_infinity_rejected(server):
    # Python's json module emits NaN/Infinity literals; the server must not accept them as amounts.
    for literal in ("NaN", "Infinity", "-Infinity"):
        server.send_raw(
            '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"trigger_refund","arguments":'
            '{"customer_id":"CUST-10003","reason":"Customer charged twice","amount":%s}}}' % literal
        )
        resp = server.recv()
        assert resp["error"]["code"] in (INVALID_PARAMS, PARSE_ERROR), literal


def test_missing_arguments_lists_all_required_fields(server):
    resp = server.call("trigger_refund", None)
    assert resp["error"]["code"] == INVALID_PARAMS
    assert {e["field"] for e in resp["error"]["data"]["errors"]} == {"customer_id", "amount", "reason"}
    assert all(e["type"] == "missing" for e in resp["error"]["data"]["errors"])


def test_validation_errors_do_not_echo_input(server):
    secret = "CUST-<script>alert(1)</script>"
    resp = server.call("get_customer_record", {"customer_id": secret})
    assert secret not in json.dumps(resp)


def test_unknown_tool_is_invalid_params(server):
    assert server.call("drop_tables", {})["error"]["code"] == INVALID_PARAMS


def test_unknown_method(server):
    assert server.request("tools/explode")["error"]["code"] == METHOD_NOT_FOUND


# --------------------------------------------------------------------------- framing errors
def test_parse_error_and_invalid_request_framing(server):
    server.send_raw("{not json")
    assert server.recv()["error"] == {"code": PARSE_ERROR, "message": "Parse error"}
    server.send_raw('{"id": 5, "hello": "world"}')
    resp = server.recv()
    assert resp["error"]["code"] == INVALID_REQUEST and resp["id"] == 5
    server.send_raw('[{"jsonrpc":"2.0","id":6,"method":"ping"}]')
    assert server.recv()["error"]["code"] == INVALID_REQUEST
    # the session is still healthy afterwards
    assert server.request("ping")["result"] == {}


# --------------------------------------------------------------------------- stdio isolation
def test_stray_prints_never_reach_stdout():
    """Inject every kind of rogue output into a tool handler and prove stdout stays pure."""
    rogue = """
        import os, subprocess, sys
        import task1_mcp_server.data as data
        _orig_get = data.CustomerStore.get
        def noisy_get(self, cid):
            print("ROGUE print()")
            sys.stdout.write("ROGUE sys.stdout.write\\n")
            os.write(1, b"ROGUE os.write(1)\\n")
            subprocess.run(["echo", "ROGUE child process"])
            import logging; logging.getLogger("x").warning("ROGUE logging")
            return _orig_get(self, cid)
        data.CustomerStore.get = noisy_get
    """
    h = StdioHarness(rogue)
    h.initialize()
    assert h.call("get_customer_record", {"customer_id": "CUST-10001"})["result"]["isError"] is False
    stderr = h.close()
    h.assert_stdout_pure()
    assert all(b"ROGUE" not in line for line in h.raw_stdout)
    for marker in (
        "ROGUE print()",
        "ROGUE sys.stdout.write",
        "ROGUE os.write(1)",
        "ROGUE child process",
        "ROGUE logging",
    ):
        assert marker in stderr, f"{marker!r} should have been diverted to stderr"


# --------------------------------------------------------------------------- SDK interop
async def test_official_sdk_client_interop():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.shared.exceptions import MCPError

    params = StdioServerParameters(command=sys.executable, args=["-m", "task1_mcp_server"], cwd=str(ROOT))
    with open(os.devnull, "w") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {t.name for t in tools.tools} == {"get_customer_record", "trigger_refund"}
            ok = await session.call_tool("get_customer_record", {"customer_id": "CUST-10001"})
            assert ok.is_error is False
            with pytest.raises(MCPError) as exc_info:
                await session.call_tool("trigger_refund", {"customer_id": "bad", "amount": -1, "reason": "x"})
            assert exc_info.value.code == INVALID_PARAMS


# --------------------------------------------------------------------------- iteration 2 (spec research)
def test_tool_annotations_and_output_schema(server):
    tools = {t["name"]: t for t in server.request("tools/list")["result"]["tools"]}
    assert [t for t in tools] == ["get_customer_record", "trigger_refund"]  # deterministic order
    assert tools["get_customer_record"]["annotations"]["readOnlyHint"] is True
    assert tools["trigger_refund"]["annotations"]["destructiveHint"] is True
    assert tools["trigger_refund"]["annotations"]["idempotentHint"] is False
    out = tools["trigger_refund"]["outputSchema"]
    assert out["type"] == "object" and "refund_id" in out["required"]
    for t in tools.values():
        assert "$ref" not in json.dumps(t["inputSchema"])


def test_invalid_utf8_is_parse_error(server):
    server.proc.stdin.write(b'\xff\xfe{"jsonrpc":"2.0","id":1,"method":"ping"}\n')
    server.proc.stdin.flush()
    assert server.recv()["error"]["code"] == PARSE_ERROR


def test_malformed_notification_gets_no_reply(server):
    server.send_raw('{"method":"notifications/whatever","bogus":true}')
    # The very next response must belong to the ping, i.e. nothing was emitted for the notification.
    assert server.request("ping")["result"] == {}


def test_validation_errors_as_tool_results_mode():
    """MCP 2025-11-25+ (SEP-1303) recommends isError results for bad arguments; supported via a toggle."""
    h = StdioHarness("import os; os.environ['MCP_VALIDATION_ERRORS'] = 'tool'")
    h.initialize()
    result = h.call("trigger_refund", {"customer_id": "CUST-1", "amount": -1, "reason": "short"})["result"]
    h.close()
    h.assert_stdout_pure()
    assert result["isError"] is True
    assert {e["field"] for e in result["structuredContent"]["errors"]} == {"customer_id", "amount", "reason"}


def test_modern_protocol_discover_path():
    """2026-07-28 stateless era: no initialize handshake, per-request _meta envelope."""
    meta = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
    }
    h = StdioHarness()
    disc = h.request("server/discover", {"_meta": meta})
    assert "2026-07-28" in disc["result"]["supportedVersions"]
    bad = h.request("tools/call", {"name": "get_customer_record", "arguments": {"customer_id": "x"}, "_meta": meta})
    good = h.request(
        "tools/call", {"name": "get_customer_record", "arguments": {"customer_id": "CUST-10001"}, "_meta": meta}
    )
    h.close()
    h.assert_stdout_pure()
    assert bad["error"]["code"] == INVALID_PARAMS
    assert good["result"]["structuredContent"]["customer_id"] == "CUST-10001"


def test_exits_promptly_on_stdin_eof():
    import time

    h = StdioHarness()
    h.initialize()
    start = time.monotonic()
    h.close()
    assert time.monotonic() - start < 3
    assert h.proc.returncode == 0


# --------------------------------------------------------------------------- iteration 3 (adversarial review regressions)
@pytest.mark.parametrize(
    "frame",
    [
        '{"jsonrpc":"2.0","id":"\\ud800","method":"ping"}',
        '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"\\ud800","arguments":{}}}',
        '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"get_customer_record","arguments":{"customer_id":"\\udfff"}}}',
    ],
)
def test_lone_surrogates_rejected_and_server_survives(server, frame):
    server.send_raw(frame)
    assert server.recv()["error"]["code"] == INVALID_REQUEST
    assert server.request("ping")["result"] == {}
    assert server.proc.poll() is None


@pytest.mark.parametrize("bad_id", ["true", "1.0", "{}", "null", "[1]"])
def test_invalid_request_ids_get_an_error_not_silence(server, bad_id):
    server.send_raw(
        '{"jsonrpc":"2.0","id":%s,"method":"tools/call","params":{"name":"trigger_refund","arguments":'
        '{"customer_id":"CUST-10001","amount":1.0,"reason":"should never execute"}}}' % bad_id
    )
    assert server.recv()["error"]["code"] == INVALID_REQUEST
    # and the refund must not have happened
    rec = server.call("get_customer_record", {"customer_id": "CUST-10001"})["result"]["structuredContent"]
    assert rec["refunded_total"] == 0.0


def test_oversized_frame_is_bounded_and_answered(server):
    from task1_mcp_server.transport import MAX_LINE_BYTES

    chunk = b"x" * 1_000_000
    for _ in range(MAX_LINE_BYTES // len(chunk) + 3):
        server.proc.stdin.write(chunk)
    server.proc.stdin.write(b"\n")
    server.proc.stdin.flush()
    assert server.recv(timeout=20)["error"] == {"code": INVALID_REQUEST, "message": "Request too large"}
    assert server.request("ping")["result"] == {}


def test_huge_integer_is_invalid_request_not_parse_error(server):
    server.send_raw('{"jsonrpc":"2.0","id":1,"method":"ping","params":{"x":%s}}' % ("9" * 5000))
    assert server.recv()["error"]["code"] == INVALID_REQUEST


@pytest.mark.parametrize("reason", ["​" * 10, "\x00" * 10, "a​b​c​d​e", " " * 12, "a b c d e f g"])
def test_reason_must_have_real_content(server, reason):
    resp = server.call("trigger_refund", {"customer_id": "CUST-10001", "amount": 1.0, "reason": reason})
    assert resp["error"]["code"] == INVALID_PARAMS


def test_bad_log_level_does_not_crash():
    h = StdioHarness("import os; os.environ['MCP_LOG_LEVEL'] = 'LOUD'")
    h.initialize()
    assert h.request("ping")["result"] == {}
    h.close()


# --------------------------------------------------------------------------- iteration 4 (peer benchmark)
def test_idempotency_key_prevents_double_refund(server):
    args = {
        "customer_id": "CUST-10003",
        "amount": 100.0,
        "reason": "Duplicate charge on invoice 7",
        "idempotency_key": "refund-abc12345",
    }
    first = server.call("trigger_refund", args)["result"]["structuredContent"]
    retry = server.call("trigger_refund", args)["result"]["structuredContent"]
    assert first["refund_id"] == retry["refund_id"] and retry["replayed"] is True and first["replayed"] is False
    rec = server.call("get_customer_record", {"customer_id": "CUST-10003"})["result"]["structuredContent"]
    assert rec["refunded_total"] == 100.0  # charged once
    conflict = server.call("trigger_refund", {**args, "amount": 5.0})["result"]
    assert conflict["isError"] is True and "different arguments" in conflict["content"][0]["text"]


@pytest.mark.parametrize("key", ["short", "x" * 65, "has space!", ""])
def test_idempotency_key_format_validated(server, key):
    args = {
        "customer_id": "CUST-10003",
        "amount": 1.0,
        "reason": "Duplicate charge on invoice 7",
        "idempotency_key": key,
    }
    assert server.call("trigger_refund", args)["error"]["code"] == INVALID_PARAMS


NOISY_DEPENDENCY = """
import importlib.abc, sys
class _Noisy(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "pydantic":
            print("ROGUE import-time print from a dependency")
        return None
sys.meta_path.insert(0, _Noisy())
"""


def test_import_time_prints_cannot_reach_the_wire():
    """`python -m task1_mcp_server` claims fd 1 before importing any third-party package."""
    h = StdioHarness(NOISY_DEPENDENCY, module="task1_mcp_server")
    h.initialize()
    stderr = h.close()
    h.assert_stdout_pure()
    assert "ROGUE import-time print" in stderr


def test_negative_control_purity_check_can_fail():
    """Negative control: importing the server module *before* claiming the wire lets the rogue print
    through - proving the purity assertion above is capable of failing."""
    h = StdioHarness(NOISY_DEPENDENCY, module="task1_mcp_server.server")
    h.initialize_raw_tolerant()
    h.close()
    assert any(b"ROGUE" in line for line in h.raw_stdout)
