"""Pre-flight token estimation for reservations.

A reservation must be computed *before* the provider tells us the real count, so
it is an estimate by design and is later reconciled to ``usage.total_tokens``.

* prompt: ~4 chars/token plus the per-message framing overhead from the OpenAI cookbook
  (3 tokens/message + 3 for the reply primer). Swap in ``tiktoken`` for exact counts
  per model; the reconcile step makes the heuristic's error self-correcting either way.
* completion: ``max_completion_tokens`` / ``max_tokens`` if given (OpenAI reserves this
  too), else a configured default.
"""

from __future__ import annotations

import json
import math
from typing import Any

CHARS_PER_TOKEN = 4
PER_MESSAGE_OVERHEAD = 3
REPLY_PRIMER = 3


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multi-part content: [{"type": "text", "text": ...}, ...]
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return json.dumps(content)


def estimate_prompt_tokens(messages: list[dict[str, Any]], tools: list | None = None) -> int:
    total = REPLY_PRIMER
    for m in messages:
        total += PER_MESSAGE_OVERHEAD + math.ceil(len(_text_of(m.get("content"))) / CHARS_PER_TOKEN)
        if m.get("name"):
            total += 1
        if m.get("tool_calls"):
            total += math.ceil(len(json.dumps(m["tool_calls"])) / CHARS_PER_TOKEN)
    if tools:
        total += math.ceil(len(json.dumps(tools)) / CHARS_PER_TOKEN)
    return total


def estimate_text_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)
