"""Task 5, reproduced: an idle-timeout middlebox kills a quiet SSE stream; heartbeats keep it alive.

client ──► idle_proxy (kills connections idle > 1 s) ──► Task 3 guardrail ──► provider that "thinks" 2.5 s
"""

from __future__ import annotations

import asyncio
import socket
import threading

import httpx
import pytest

from task3_llm_guardrail.gateway import GuardSettings, create_app
from task3_llm_guardrail.mock_llm import build as build_mock
from task5_zero_trust.idle_proxy import serve

from .conftest import run_uvicorn


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class IdleProxy:
    def __init__(self, target_port: int, idle_timeout: float):
        self.port, self.stats = _free_port(), {"idle_kills": 0}
        self._loop = asyncio.new_event_loop()
        started = threading.Event()

        def run() -> None:
            asyncio.set_event_loop(self._loop)
            self._server = self._loop.run_until_complete(
                serve(self.port, "127.0.0.1", target_port, idle_timeout, self.stats)
            )
            started.set()
            self._loop.run_forever()

        threading.Thread(target=run, daemon=True).start()
        started.wait(5)

    def close(self) -> None:
        async def shutdown() -> None:
            self._server.close()
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        asyncio.run_coroutine_threadsafe(shutdown(), self._loop).result(5)
        self._loop.call_soon_threadsafe(self._loop.stop)


@pytest.mark.parametrize("heartbeat", [None, 0.3])
def test_idle_timeout_middlebox_vs_heartbeats(heartbeat):
    mock_app, _ = build_mock()
    with run_uvicorn(mock_app) as m, run_uvicorn(create_app(GuardSettings(f"{m}/v1", heartbeat_s=heartbeat))) as gw:
        proxy = IdleProxy(int(gw.rsplit(":", 1)[1]), idle_timeout=1.0)
        body = {"stream": True, "messages": [{"role": "user", "content": "the answer is 42"}], "mock_ttft_ms": 2500}
        try:
            try:
                text = httpx.post(f"http://127.0.0.1:{proxy.port}/v1/chat/completions", json=body, timeout=10).text
                outcome = "complete" if text.rstrip().endswith("data: [DONE]") else "truncated"
            except httpx.TransportError:
                outcome = "dropped"
        finally:
            proxy.close()
    if heartbeat is None:
        # the "intermittent hang/drop" from the Task 5 scenario: killed by the middlebox while the model thinks
        assert outcome in ("dropped", "truncated") and proxy.stats["idle_kills"] >= 1
    else:
        assert outcome == "complete" and proxy.stats["idle_kills"] == 0
