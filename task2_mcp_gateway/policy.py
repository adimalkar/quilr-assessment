"""Tool-level authorization policy.

The check runs on a *normalised* tool name so look-alike names can't bypass it:
``ADMIN_reset_key``, full-width ``ａｄｍｉｎ_reset_key``, or ``admin​_reset_key``
(zero-width space) are all treated as admin tools. If the downstream server would
normalise differently, we still fail closed: anything that *looks* admin is admin.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .auth import Principal

ADMIN_PREFIX = "admin_"
# MCP 2025-11-25 tool-name guidance: 1-128 chars of [A-Za-z0-9_.-]
VALID_TOOL_NAME = re.compile(r"[A-Za-z0-9_.\-]{1,128}")


def normalise_tool_name(name: str) -> str:
    # NFKC folds compatibility forms (full-width, ligatures); drop format chars (zero-width, bidi
    # controls); casefold handles case and things like German sharp s.
    folded = unicodedata.normalize("NFKC", name)
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Cf")
    return folded.strip().casefold()


def is_admin_tool(name: str) -> bool:
    return normalise_tool_name(name).startswith(ADMIN_PREFIX)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    code: str = "allowed"


def authorize_tool_call(principal: Principal, tool_name: str) -> Decision:
    # Admin check first, on the normalised name, so look-alikes get the authorization error.
    admin_tool = is_admin_tool(tool_name)
    if admin_tool and not principal.is_admin:
        return Decision(False, f"tool requires admin role (caller role: {principal.role})", "forbidden_tool")
    if not VALID_TOOL_NAME.fullmatch(tool_name):
        # Fail closed on names a downstream might normalise into something we didn't evaluate.
        return Decision(False, "tool name outside the MCP-recommended character set", "invalid_tool_name")
    return Decision(True, "admin tool, admin role" if admin_tool else "non-admin tool")


def tool_visible(principal: Principal, tool_name: str) -> bool:
    """Used by the optional tools/list filter (least-privilege discovery)."""
    return principal.is_admin or not is_admin_tool(tool_name)
