"""Bearer-token authentication for the MCP gateway.

Tokens are HS256 JWTs carrying a ``role`` claim (``admin`` | ``viewer``). Signature,
``exp`` and (when configured) ``iss`` / ``aud`` are verified; ``alg`` is pinned so
``alg: none`` / algorithm-confusion tokens are rejected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum

import jwt


class Role(StrEnum):
    ADMIN = "admin"
    VIEWER = "viewer"


@dataclass(frozen=True)
class Principal:
    subject: str
    role: Role

    @property
    def is_admin(self) -> bool:
        return self.role is Role.ADMIN


class AuthError(Exception):
    """Authentication failed. ``message`` is safe to return to the client."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class TokenVerifier:
    ALGORITHMS = ["HS256"]

    def __init__(self, secret: str, *, issuer: str | None = None, audience: str | None = None, leeway: int = 30):
        if len(secret) < 32:
            raise ValueError("JWT secret must be at least 32 characters")
        self._secret = secret
        self._issuer = issuer
        self._audience = audience
        self._leeway = leeway

    def verify_header(self, authorization: str | None) -> Principal:
        if not authorization:
            raise AuthError("Missing Authorization header")
        scheme, _, token = authorization.strip().partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise AuthError("Authorization header must be 'Bearer <token>'")
        return self.verify(token.strip())

    def verify(self, token: str) -> Principal:
        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=self.ALGORITHMS,
                issuer=self._issuer,
                audience=self._audience,
                leeway=self._leeway,
                options={"require": ["exp", "sub", "role"], "verify_aud": self._audience is not None},
            )
        except jwt.ExpiredSignatureError:
            raise AuthError("Token expired") from None
        except jwt.PyJWTError:
            # Deliberately vague: don't tell an attacker *which* check failed.
            raise AuthError("Invalid token") from None

        try:
            role = Role(claims["role"])
        except (ValueError, TypeError):
            raise AuthError("Invalid token") from None
        subject = claims["sub"]
        if not isinstance(subject, str) or not subject:
            raise AuthError("Invalid token")
        return Principal(subject=subject, role=role)

    def mint(self, subject: str, role: Role | str, ttl_seconds: int = 3600) -> str:
        """Issue a token (dev/test helper - production tokens come from your IdP)."""
        now = int(time.time())
        claims = {"sub": subject, "role": str(Role(role)), "iat": now, "exp": now + ttl_seconds}
        if self._issuer:
            claims["iss"] = self._issuer
        if self._audience:
            claims["aud"] = self._audience
        return jwt.encode(claims, self._secret, algorithm="HS256")
