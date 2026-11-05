"""Exact incremental PII redaction over a text stream with a provably minimal holdback.

The problem
-----------
LLM output arrives as tiny deltas (``"Contact jo"``, ``"hn@exam"``, ``"ple.com"``). A
per-chunk regex misses PII split across chunks; buffering the whole response
destroys time-to-first-token. Fixed windows (NeMo ``chunk_size``/``context_size``,
sentence buffering) either over-buffer or still leak when an entity straddles the
window edge.

The algorithm
-------------
Keep an unreleased buffer ``buf``. After each delta compute the *hold point*

    h = min{ i : buf[i:] could still be the beginning of a PII match }

using the ``regex`` module's ``fullmatch(buf, pos=i, partial=True)``: it succeeds iff
the remaining text is a (complete or partial) match that more input could extend.
Everything before ``h`` is *final*:

* no match can start at ``p < h`` and reach past the end of the buffer (else
  ``buf[p:]`` would be a match prefix and ``p >= h``), so
* every match found in ``buf[:h]`` already has its final extent, and
* every position in ``buf[:h]`` that does not start a match never will.

So we redact the matches in ``buf[:h]``, release that text, and keep only ``buf[h:]``,
which is the shortest possible suffix. In practice ``h`` is the start of the last
word or number, so latency is one token, not one sentence.

Guarantees (property-tested in ``tests/test_task3_guardrail.py``):

* **Chunking invariance**: for *any* split of a text into deltas, the concatenated
  output equals ``redact_text(full_text)``. Streaming never leaks what batch mode would catch.
* **Bounded memory**: every pattern has a bounded length, so ``len(buf) <= MAX_MATCH_LEN``
  no matter how long the stream is; partial-match checks are O(held text).
* **Monotonic scanning**: positions ruled out stay ruled out (if ``buf[i:]`` is not a
  match prefix, no extension of it is either), so each check resumes from the hold point.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import unicodedata

import regex

REDACTED = "[REDACTED]"

# --------------------------------------------------------------------------- patterns
# Every quantifier is bounded so a match (and therefore the holdback buffer) is bounded.
# Letters/digits are Unicode-aware (\w, \d): IDN domains and full-width / non-ASCII digits are caught too.
_EMAIL = (
    r"(?<![\w.%+\-])"  # don't start mid-token
    r"[\w.%+\-]{1,64}@"  # local part (RFC 5321: <= 64)
    r"(?:[^\W_](?:[\w\-]{0,61}[^\W_])?\.){1,8}"  # labels (IDN-friendly)
    r"[^\W\d_]{2,63}"  # TLD
    r"(?![\w\-])"
)
_SEP = r"[ \-.\u2013\u2014]"  # space, hyphen, dot, en/em dash
# US SSN with one consistent separator (named backref: numbered refs break once patterns are combined).
# Presidio-style invalid ranges excluded: area 000/666, group 00, serial 0000. Only digits bound the
# match, so "SSN-123-45-6789" and "123-45-6789-x" are caught.
_SSN = rf"(?<!\d)(?!000|666)\d{{3}}(?P<ssn_sep>{_SEP})(?!00)\d{{2}}(?P=ssn_sep)(?!0000)\d{{4}}(?!\d)"
# Unseparated 9-digit SSNs are only flagged with an explicit nearby label (bare 9-digit numbers are far
# too often order ids). The label is part of the match and is put back by the replacer - deliberately
# *not* a variable-length lookbehind, which `regex` mishandles in partial-matching mode (spurious partials).
_SSN_LABELLED = (
    r"(?P<ssn_label>(?i:\bssn|\bsocial[ \t]+security(?:[ \t]+(?:number|no\.?|num|\#))?)[\s:#=\-]{0,4}"
    r"(?i:(?:is|was|of)[ \t]{1,3})?)"
    r"(?P<ssn_digits>(?!000|666)\d{3}(?!00)\d{2}(?!0000)\d{4})(?!\d)"
)
# Payment cards: 13-19 contiguous digits, or 4-4-4-(1..7) / Amex 4-6-5 groups with any separators.
# Every candidate must pass Luhn, which is what keeps false positives down.
_CARD = (
    r"(?<!\d)"
    r"(?:\d{13,19}"
    rf"|\d{{4}}{_SEP}\d{{4}}{_SEP}\d{{4}}{_SEP}\d{{1,7}}"
    rf"|\d{{4}}{_SEP}\d{{6}}{_SEP}\d{{4,5}})"
    r"(?!\d)"
)

PII_PATTERN = regex.compile(
    rf"(?P<email>{_EMAIL})|(?P<ssn>{_SSN})|(?P<ssn_labelled>{_SSN_LABELLED})|(?P<card>{_CARD})"
)
# Hold-point detection uses each alternative compiled on its own: "could buf[i:] still begin a match of
# A|B|C" is exactly "could it begin a match of A, B or C" - and it keeps each partial check simple.
HOLD_PATTERNS = tuple(regex.compile(p) for p in (_EMAIL, _SSN, _SSN_LABELLED, _CARD))
MAX_MATCH_LEN = 64 + 1 + 8 * 64 + 63  # longest possible email; cards/SSNs are shorter
# Raw left context kept for lookbehinds/word boundaries (patterns only look back 1 char; slack is cheap).
LOOKBEHIND_CONTEXT = 8


def luhn_valid(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = unicodedata.decimal(ch)  # works for full-width / Arabic-Indic digits too
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _replacement(match: regex.Match) -> str:
    if match.group("ssn_labelled") is not None:
        return match.group("ssn_label") + REDACTED  # keep the label, redact the number
    if match.group("card") is not None:
        digits = "".join(ch for ch in match.group() if ch.isdigit())
        if not luhn_valid(digits):
            return match.group()  # order numbers, timestamps, phone-ish runs pass through
    return REDACTED


def redact_text(text: str) -> tuple[str, int]:
    """Batch reference implementation. Returns (redacted_text, number_of_redactions)."""
    count = 0

    def repl(m: regex.Match) -> str:
        nonlocal count
        out = _replacement(m)
        count += out != m.group()
        return out

    return PII_PATTERN.sub(repl, text), count


# --------------------------------------------------------------------------- streaming
@dataclass
class StreamRedactor:
    """Feed deltas with :meth:`feed`, call :meth:`flush` at end of stream."""

    pattern: regex.Pattern = PII_PATTERN
    replacement: Callable[[regex.Match], str] = _replacement
    hold_patterns: tuple[regex.Pattern, ...] = HOLD_PATTERNS
    _buf: str = ""
    _ctx: str = ""  # last released char(s): lookbehind context for the next scan
    redactions: int = 0
    max_buffered: int = 0  # high-water mark, exported for metrics/tests

    def feed(self, delta: str) -> str:
        if not delta:
            return ""
        self._buf += delta
        self.max_buffered = max(self.max_buffered, len(self._buf))
        return self._release(self._hold_point())

    def flush(self) -> str:
        """End of stream: nothing can grow any more, release everything."""
        return self._release(len(self._buf))

    @property
    def pending(self) -> str:
        return self._buf

    # ------------------------------------------------------------------ internals
    def _hold_point(self) -> int:
        text, offset = self._ctx + self._buf, len(self._ctx)
        # Only the last MAX_MATCH_LEN chars can be a match prefix (patterns are bounded).
        start = max(offset, len(text) - MAX_MATCH_LEN)
        matchers = [p.fullmatch for p in self.hold_patterns]
        for i in range(start, len(text)):
            for fullmatch in matchers:
                if fullmatch(text, pos=i, partial=True) is not None:
                    return i - offset
        return len(self._buf)

    def _release(self, cut: int) -> str:
        if cut <= 0:
            return ""
        text, offset = self._ctx + self._buf, len(self._ctx)
        end = offset + cut
        out: list[str] = []
        pos = offset
        for m in self.pattern.finditer(text, pos=offset, endpos=len(text)):
            if m.start() >= end:
                break
            # Matches starting before the cut are final (see module docstring) even if they
            # extend past it; consume them whole and move the cut to their end.
            out.append(text[pos : m.start()])
            replaced = self.replacement(m)
            self.redactions += replaced != m.group()
            out.append(replaced)
            pos = m.end()
            end = max(end, pos)
        out.append(text[pos:end])
        released_raw = text[offset:end]
        self._buf = text[end:]
        # Keep enough raw (pre-redaction) left context for the longest lookbehind - batch mode sees it too.
        self._ctx = (self._ctx + released_raw)[-LOOKBEHIND_CONTEXT:]
        return "".join(out)
