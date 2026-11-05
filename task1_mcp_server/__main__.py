"""``python -m task1_mcp_server`` - the recommended entry point.

Claims stdout *before* importing the MCP SDK, pydantic, anyio, ... so that even an import-time
``print()`` in a dependency lands on stderr instead of corrupting the JSON-RPC stream.
(``python -m task1_mcp_server.server`` still works, but imports its dependencies first.)
"""

from task1_mcp_server.stdio_guard import claim_stdio, configure_logging

configure_logging()
claim_stdio()

from task1_mcp_server.server import main  # noqa: E402  (deliberately after the claim)

main()
