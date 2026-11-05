"""Dev helper: mint gateway JWTs so the gateway can be exercised by hand.

    export GATEWAY_JWT_SECRET=$(openssl rand -hex 32)
    uv run python -m task2_mcp_gateway.devtoken --role viewer --sub bob
    uv run python -m task2_mcp_gateway.devtoken --role admin --sub alice --ttl 600

Production tokens come from your identity provider; this exists for demos and tests only.
"""

from __future__ import annotations

import argparse
import os
import sys

from .auth import Role, TokenVerifier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mint an HS256 JWT for the MCP security gateway (dev only).")
    parser.add_argument("--role", choices=[r.value for r in Role], required=True)
    parser.add_argument("--sub", default="dev-user", help="subject (user id) claim")
    parser.add_argument("--ttl", type=int, default=3600, help="lifetime in seconds")
    parser.add_argument(
        "--secret", default=os.environ.get("GATEWAY_JWT_SECRET"), help="signing secret (default: $GATEWAY_JWT_SECRET)"
    )
    parser.add_argument("--issuer", default=os.environ.get("GATEWAY_JWT_ISSUER"))
    parser.add_argument("--audience", default=os.environ.get("GATEWAY_JWT_AUDIENCE"))
    args = parser.parse_args(argv)
    if not args.secret:
        parser.error("set GATEWAY_JWT_SECRET or pass --secret")
    try:
        verifier = TokenVerifier(args.secret, issuer=args.issuer, audience=args.audience)
    except ValueError as exc:
        parser.error(str(exc))
    print(verifier.mint(args.sub, args.role, ttl_seconds=args.ttl))
    return 0


if __name__ == "__main__":
    sys.exit(main())
