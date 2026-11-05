"""Incremental Server-Sent Events decoder (WHATWG rules), modelled on openai-python's
``SSEDecoder`` / ``_SSELineDecoder``.

* Network chunks are arbitrary byte slices: a multi-byte UTF-8 char, a line, or a
  CRLF pair may be split across them. An incremental UTF-8 decoder and a pending
  "last char was CR" flag handle both.
* Lines end with CR, LF or CRLF. ``data:`` lines accumulate (joined by ``\\n``), a blank
  line dispatches the event, ``:`` lines are comments (keep-alives).
"""

from __future__ import annotations

import codecs
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass
class SSEEvent:
    data: str
    event: str | None = None
    id: str | None = None
    retry: int | None = None


@dataclass
class SSEDecoder:
    max_event_bytes: int = 1_048_576
    _decoder: codecs.IncrementalDecoder = field(
        default_factory=lambda: codecs.getincrementaldecoder("utf-8")(errors="strict")
    )
    _partial_line: str = ""
    _skip_lf: bool = False  # previous chunk ended in CR; a leading LF is part of that CRLF
    _data: list[str] = field(default_factory=list)
    _event: str | None = None
    _id: str | None = None
    _retry: int | None = None
    _size: int = 0

    def feed(self, chunk: bytes) -> Iterator[SSEEvent]:
        text = self._decoder.decode(chunk)
        for line in self._lines(text):
            event = self._process_line(line)
            if event is not None:
                yield event

    def close(self) -> Iterator[SSEEvent]:
        text = self._decoder.decode(b"", final=True)
        for line in self._lines(text):
            if (event := self._process_line(line)) is not None:
                yield event
        if self._partial_line:
            line, self._partial_line = self._partial_line, ""
            if (event := self._process_line(line)) is not None:
                yield event
        if (event := self._process_line("")) is not None:  # dispatch a trailing event without blank line
            yield event

    # ------------------------------------------------------------------ internals
    def _lines(self, text: str) -> Iterator[str]:
        if self._skip_lf and text.startswith("\n"):
            text = text[1:]
        self._skip_lf = False
        buf = self._partial_line + text
        start = 0
        i = 0
        n = len(buf)
        while i < n:
            ch = buf[i]
            if ch == "\r" or ch == "\n":
                yield buf[start:i]
                if ch == "\r":
                    if i + 1 < n:
                        if buf[i + 1] == "\n":
                            i += 1
                    else:
                        self._skip_lf = True
                start = i + 1
            i += 1
        self._partial_line = buf[start:]
        if len(self._partial_line) > self.max_event_bytes:
            raise ValueError("SSE line exceeds maximum size")

    def _process_line(self, line: str) -> SSEEvent | None:
        if not line:
            if not self._data and self._event is None:
                return None
            event = SSEEvent(data="\n".join(self._data), event=self._event, id=self._id, retry=self._retry)
            self._data, self._event, self._retry, self._size = [], None, None, 0
            return event
        if line.startswith(":"):
            return None
        name, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if name == "data":
            self._size += len(value)
            if self._size > self.max_event_bytes:
                raise ValueError("SSE event exceeds maximum size")
            self._data.append(value)
        elif name == "event":
            self._event = value
        elif name == "id" and "\0" not in value:
            self._id = value
        elif name == "retry" and value.isdigit():
            self._retry = int(value)
        return None


def encode_event(data: str, event: str | None = None) -> bytes:
    out = []
    if event:
        out.append(f"event: {event}")
    out.extend(f"data: {line}" for line in data.split("\n"))
    return ("\n".join(out) + "\n\n").encode()
