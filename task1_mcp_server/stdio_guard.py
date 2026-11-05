"""Stdout isolation for the stdio transport. **Stdlib imports only** - this module must be importable
before any third-party package, so the wire is claimed before a dependency's import-time ``print()``
(or a C extension's ``printf``) could run. See ``__main__.py``.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import BinaryIO

_claimed: tuple[BinaryIO, BinaryIO] | None = None


def configure_logging() -> None:
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


def claim_stdio() -> tuple[BinaryIO, BinaryIO]:
    """Reserve the real stdout for JSON-RPC and divert everything else to stderr (idempotent).

    Returns binary handles for the wire (stdin, stdout).
    """
    global _claimed
    if _claimed is not None:
        return _claimed
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
    _claimed = (os.fdopen(wire_in_fd, "rb"), os.fdopen(wire_out_fd, "wb"))
    return _claimed
