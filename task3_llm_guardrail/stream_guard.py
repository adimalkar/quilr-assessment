"""Re-frames an OpenAI-compatible ``chat.completion.chunk`` stream through per-field redactors.

**Every string in a choice is redacted**, whatever its key: ``delta.content``, ``refusal``,
``reasoning(_content)``, ``tool_calls[].function.arguments``, legacy ``function_call.arguments``,
list-valued ``content[].text``, ``audio.transcript``, legacy ``text``, and any field a provider
invents tomorrow. An allowlist of "fields we know carry text" is exactly how PII leaks (LiteLLM's
Presidio hook only masks ``content``). Only short structural identifiers (``role``, ``type``, ``id``,
``name``, ...) are exempt from *streaming* holdback - they still get one-shot redaction - so tool
names and ids are never split across chunks.

Each string field gets its own :class:`StreamRedactor`, keyed by ``(choice index, JSON path)``.
List elements are addressed by their ``index`` field when present (tool calls), else by ``id``,
else by position within the chunk - so parallel tool calls, ``n > 1`` choices and multi-part content
never bleed into each other's buffers.

* ``logprobs`` is *dropped*: it carries raw token text/bytes that would bypass redaction.
* Released text rides in the envelope (``id``/``model``/``created``) of the chunk that completed it.
* Chunks whose text is entirely held back are suppressed unless they carry structure
  (role, tool-call id/name, finish_reason, usage).
* Held text is flushed into the chunk carrying ``finish_reason`` for that choice, or into a
  synthetic chunk before ``[DONE]``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .redactor import StreamRedactor, redact_text

# Short identifier-like values: never held back (would split names/ids across chunks), but still
# passed through one-shot redaction so a PII-shaped value can't sneak out under these keys either.
STRUCTURAL_KEYS = frozenset({"role", "type", "id", "name", "index", "finish_reason", "object", "model"})
DROPPED_KEYS = frozenset({"logprobs"})

PathElem = str | tuple[str, Any]


def _elem_key(item: Any, position: int) -> tuple[str, Any]:
    if isinstance(item, dict):
        if isinstance(item.get("index"), int):
            return ("@index", item["index"])
        if isinstance(item.get("id"), str):
            return ("@id", item["id"])
    return ("@pos", position)


@dataclass
class GuardStats:
    redactions: int = 0
    max_buffered: int = 0
    chunks_in: int = 0
    chunks_out: int = 0


@dataclass
class ChunkGuard:
    _redactors: dict[tuple, StreamRedactor] = field(default_factory=dict)
    _finished: set[Any] = field(default_factory=set)
    _last_envelope: dict[str, Any] | None = None
    _oneshot_redactions: int = 0
    stats: GuardStats = field(default_factory=GuardStats)

    def _r(self, key: tuple) -> StreamRedactor:
        r = self._redactors.get(key)
        if r is None:
            r = self._redactors[key] = StreamRedactor()
        return r

    # ------------------------------------------------------------------ tree walk
    def _walk(self, choice_idx: Any, node: Any, path: tuple) -> tuple[Any, bool]:
        """Redact all strings under ``node``. Returns (new_node, emitted_something)."""
        if isinstance(node, str):
            safe = self._r((choice_idx, path)).feed(node)
            return safe, bool(safe)
        if isinstance(node, dict):
            emitted = False
            out: dict[str, Any] = {}
            for key, value in node.items():
                if key in DROPPED_KEYS:
                    continue
                if key in STRUCTURAL_KEYS:
                    if isinstance(value, str) and value:
                        redacted, n = redact_text(value)
                        self._oneshot_redactions += n
                        value = redacted
                        emitted = emitted or key not in ("type", "object", "model", "finish_reason")
                    elif key == "index" or value is None:
                        pass
                    out[key] = value
                    continue
                new_value, child_emitted = self._walk(choice_idx, value, (*path, key))
                emitted |= child_emitted
                if isinstance(value, str) and not new_value and len(path) <= 2:
                    continue  # choice/delta-level text field fully held back: omit it this chunk
                out[key] = new_value
            return out, emitted
        if isinstance(node, list):
            emitted = False
            items = []
            for pos, item in enumerate(node):
                new_item, child_emitted = self._walk(choice_idx, item, (*path, _elem_key(item, pos)))
                emitted |= child_emitted
                items.append(new_item)
            return items, emitted
        return node, False  # numbers, bools, null

    # ------------------------------------------------------------------ public
    def process(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """Return the chunk to emit (redacted), or None to suppress it."""
        self.stats.chunks_in += 1
        self._last_envelope = {k: chunk[k] for k in ("id", "object", "created", "model", "system_fingerprint") if k in chunk}
        keep = bool(chunk.get("usage"))
        new_choices = []
        for pos, choice in enumerate(chunk.get("choices") or []):
            if not isinstance(choice, dict):
                continue
            idx = choice.get("index", pos)
            body = {k: v for k, v in choice.items() if k not in ("index", "finish_reason")}
            new_body, emitted = self._walk(idx, body, ("choice",))
            new_choice = {"index": idx, **new_body}
            new_choice.setdefault("delta", {})
            if "finish_reason" in choice:
                new_choice["finish_reason"] = choice["finish_reason"]
            if choice.get("finish_reason"):
                self._flush_choice(idx, new_choice)
                self._finished.add(idx)
                emitted = True
            keep |= emitted
            new_choices.append(new_choice)
        self._update_stats()
        if not keep:
            return None
        out = {k: v for k, v in chunk.items() if k != "choices"}
        out["choices"] = new_choices
        self.stats.chunks_out += 1
        return out

    def finish(self) -> dict[str, Any] | None:
        """End of upstream stream: flush whatever is still held in a synthetic chunk."""
        choices = []
        for idx in sorted({key[0] for key in self._redactors} - self._finished, key=str):
            choice: dict[str, Any] = {"index": idx, "delta": {}}
            if self._flush_choice(idx, choice):
                choice["finish_reason"] = None
                choices.append(choice)
        self._update_stats()
        if not choices:
            return None
        envelope = copy.deepcopy(self._last_envelope or {"object": "chat.completion.chunk"})
        envelope["choices"] = choices
        self.stats.chunks_out += 1
        return envelope

    # ------------------------------------------------------------------ internals
    def _flush_choice(self, idx: Any, choice: dict[str, Any]) -> bool:
        flushed = False
        for key, r in self._redactors.items():
            if key[0] != idx:
                continue
            tail = r.flush()
            if tail:
                _append_at(choice, key[1][1:], tail)  # path[0] is the "choice" root marker
                flushed = True
        return flushed

    def _update_stats(self) -> None:
        self.stats.redactions = self._oneshot_redactions + sum(r.redactions for r in self._redactors.values())
        self.stats.max_buffered = max([self.stats.max_buffered, *(r.max_buffered for r in self._redactors.values())])


def _append_at(root: dict[str, Any], path: tuple, tail: str) -> None:
    """Append ``tail`` to the string at ``path`` inside ``root``, creating containers as needed."""
    node: Any = root
    for depth, elem in enumerate(path):
        last = depth == len(path) - 1
        if isinstance(elem, tuple):  # list element
            kind, ident = elem
            target = None
            for item in node:
                if (kind == "@index" and isinstance(item, dict) and item.get("index") == ident) or (
                    kind == "@id" and isinstance(item, dict) and item.get("id") == ident
                ):
                    target = item
                    break
            if kind == "@pos" and isinstance(ident, int) and ident < len(node):
                target = node[ident]
            if target is None:
                target = {"index": ident} if kind == "@index" else ({"id": ident} if kind == "@id" else {})
                node.append(target)
            node = target
        else:
            if last:
                node[elem] = (node.get(elem) or "") + tail
                return
            nxt = path[depth + 1]
            node = node.setdefault(elem, [] if isinstance(nxt, tuple) else {})


def redact_completion(body: dict[str, Any]) -> int:
    """Non-streaming responses: redact every string in every choice, in place. Returns redaction count."""
    total = 0

    def walk(node: Any) -> Any:
        nonlocal total
        if isinstance(node, str):
            out, n = redact_text(node)
            total += n
            return out
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items() if k not in DROPPED_KEYS}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    body["choices"] = [walk(c) for c in body.get("choices") or []]
    return total
