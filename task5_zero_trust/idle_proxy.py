"""A TCP proxy that behaves like a strict corporate egress proxy / LB: it kills connections that have
been idle (no bytes in either direction) for ``idle_timeout`` seconds, with no notice to either side.

It turns the Task 5 hypothesis "SSE sessions die on middlebox idle timeouts" into a reproducible
experiment (see tests/test_task5_idle_proxy.py and README.md in this folder):

    uv run python -m task5_zero_trust.idle_proxy --listen 18300 --target 127.0.0.1:8300 --idle 5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

logger = logging.getLogger("task5.idle_proxy")


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, touch) -> None:
    try:
        while data := await reader.read(65536):
            touch()
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        writer.close()


async def _handle(client_r, client_w, target_host: str, target_port: int, idle_timeout: float, stats: dict) -> None:
    try:
        up_r, up_w = await asyncio.open_connection(target_host, target_port)
    except OSError:
        client_w.close()
        return
    last = [time.monotonic()]

    def touch() -> None:
        last[0] = time.monotonic()

    tasks = [asyncio.create_task(_pipe(client_r, up_w, touch)), asyncio.create_task(_pipe(up_r, client_w, touch))]

    async def watchdog() -> None:
        while not all(t.done() for t in tasks):
            await asyncio.sleep(min(0.05, idle_timeout / 4))
            if time.monotonic() - last[0] > idle_timeout:
                stats["idle_kills"] += 1
                logger.warning("idle for > %.1fs: dropping connection", idle_timeout)
                for t in tasks:
                    t.cancel()
                # abort = RST-like hard drop, which is what most middleboxes do
                client_w.transport.abort()
                up_w.transport.abort()
                return

    await asyncio.gather(watchdog(), *tasks, return_exceptions=True)


async def serve(listen_port: int, target_host: str, target_port: int, idle_timeout: float, stats: dict | None = None):
    stats = stats if stats is not None else {"idle_kills": 0}
    return await asyncio.start_server(
        lambda r, w: _handle(r, w, target_host, target_port, idle_timeout, stats), "127.0.0.1", listen_port
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--listen", type=int, required=True)
    ap.add_argument("--target", required=True, help="host:port")
    ap.add_argument("--idle", type=float, default=60.0, help="idle timeout in seconds")
    args = ap.parse_args()
    host, port = args.target.rsplit(":", 1)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    async def run() -> None:
        server = await serve(args.listen, host, int(port), args.idle)
        logger.info("idle-timeout proxy :%d -> %s (idle %.1fs)", args.listen, args.target, args.idle)
        async with server:
            await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
