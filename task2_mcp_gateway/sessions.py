"""Stateless binding of downstream ``Mcp-Session-Id`` values to the authenticated subject.

A downstream session id is a bearer capability. If the gateway forwarded it verbatim,
anyone who learned it (logs, a shared proxy, a guess) could ride another user's session.

Instead the gateway hands clients ``<downstream-id>.<tag>`` where
``tag = HMAC-SHA256(key, subject ‖ 0x00 ‖ downstream-id)[:16]`` (base64url), and on every
request verifies the tag against the *caller's* JWT subject before forwarding the bare
downstream id. Properties:

* no server-side table: survives gateway restarts and works across replicas sharing the key;
* fail-closed: unknown, unsigned, or foreign ids are all "Session not found" (404), which per the
  Streamable HTTP spec tells the client to start a new session;
* the downstream id itself is opaque to the gateway; any visible-ASCII id works.
"""

from __future__ import annotations

import base64
import hashlib
import hmac


class SessionBinder:
    def __init__(self, secret: str) -> None:
        # Domain-separated from the JWT signing use of the same secret.
        self._key = hmac.new(secret.encode(), b"mcp-gateway/session-binding/v1", hashlib.sha256).digest()

    def _tag(self, subject: str, downstream_id: str) -> str:
        mac = hmac.new(self._key, subject.encode() + b"\x00" + downstream_id.encode(), hashlib.sha256).digest()
        return base64.urlsafe_b64encode(mac[:16]).rstrip(b"=").decode()

    def wrap(self, subject: str, downstream_id: str) -> str:
        return f"{downstream_id}.{self._tag(subject, downstream_id)}"

    def unwrap(self, subject: str, client_value: str) -> str | None:
        """Return the downstream id if ``client_value`` was issued to ``subject``, else None."""
        downstream_id, sep, tag = client_value.rpartition(".")
        if not sep or not downstream_id or not tag:
            return None
        if not hmac.compare_digest(tag, self._tag(subject, downstream_id)):
            return None
        return downstream_id
