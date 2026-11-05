"""Token-aware sliding-window rate limiter persisted in on-disk SQLite.

Algorithm: **sliding log with weighted entries**. Each admitted request writes one
row ``(key_hash, ts_ms, tokens)``. A tenant's usage is the sum of ``tokens`` over rows
with ``ts_ms > now - window``. This is exact (no fixed-window boundary bursts, and no
approximation as in the sliding-window-counter), and one row per *request* - not per
token - keeps it cheap (pyrate-limiter's SQLiteBucket writes a row per unit).

Reserve → reconcile → (refund)
  * Before calling a model we **reserve** ``prompt_estimate + max_tokens`` (OpenAI's own
    rule is to charge max_tokens up front). This makes the limit hard: concurrent
    requests can't all pass a check and then jointly overshoot (Kong / Envoy AI
    Gateway charge after the response, so admitted requests can blow past the limit).
  * After the response we **reconcile** the row to the provider's actual
    ``usage.total_tokens`` - usually lower, so capacity is given back.
  * If no provider produced anything we **refund** (delete) the reservation.

Concurrency: check-and-insert runs inside ``BEGIN IMMEDIATE`` so it is atomic across
threads *and processes* sharing the file (several uvicorn workers). WAL lets readers
proceed during writes; ``busy_timeout`` turns lock contention into a short wait
instead of ``SQLITE_BUSY`` errors.

Eviction: expired rows for a key are deleted inside the same transaction that reads
them, and a periodic :meth:`prune` sweeps keys that went idle.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS token_usage (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash    TEXT    NOT NULL,
    ts_ms       INTEGER NOT NULL,
    tokens      INTEGER NOT NULL CHECK (tokens >= 0),
    request_id  TEXT
);
CREATE INDEX IF NOT EXISTS idx_usage_key_ts ON token_usage (key_hash, ts_ms);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON token_usage (ts_ms);

CREATE TABLE IF NOT EXISTS tenants (
    key_hash    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    tpm_limit   INTEGER,          -- NULL -> default limit
    created_at  INTEGER NOT NULL
);
"""


def hash_key(api_key: str) -> str:
    """API keys are never stored; only their SHA-256."""
    return hashlib.sha256(api_key.encode()).hexdigest()


@dataclass(frozen=True)
class WindowState:
    limit: int
    used: int
    reset_ms: int  # ms until the oldest in-window entry expires (0 if window empty)

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)


@dataclass(frozen=True)
class Reservation:
    id: int
    key_hash: str
    tokens: int
    state: WindowState  # state *after* admitting this reservation


@dataclass(frozen=True)
class RateLimited:
    retry_after_ms: int
    state: WindowState

    @property
    def retry_after_s(self) -> int:
        return max(1, math.ceil(self.retry_after_ms / 1000))


class RequestTooLarge(Exception):
    """The request alone exceeds the tenant's per-window limit: retrying can never succeed."""

    def __init__(self, tokens: int, limit: int) -> None:
        super().__init__(f"request needs {tokens} tokens; limit is {limit} per window")
        self.tokens, self.limit = tokens, limit


class SQLiteTokenLimiter:
    def __init__(
        self,
        db_path: str | Path,
        *,
        tokens_per_window: int = 50_000,
        window_ms: int = 60_000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.db_path = str(db_path)
        self.default_limit = tokens_per_window
        self.window_ms = window_ms
        self._clock = clock
        # One connection guarded by a lock: sqlite3 objects aren't safe for concurrent use,
        # and statements here take microseconds. Cross-process safety comes from BEGIN IMMEDIATE.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")  # durable across app crashes; WAL-safe
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ time
    def now_ms(self) -> int:
        return int(self._clock() * 1000)

    # ------------------------------------------------------------------ tenants
    def add_tenant(self, api_key: str, name: str, tpm_limit: int | None = None) -> str:
        key_hash = hash_key(api_key)
        with self._lock:
            self._conn.execute(
                "INSERT INTO tenants (key_hash, name, tpm_limit, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(key_hash) DO UPDATE SET name = excluded.name, tpm_limit = excluded.tpm_limit",
                (key_hash, name, tpm_limit, self.now_ms()),
            )
        return key_hash

    def tenant(self, api_key: str) -> tuple[str, str, int] | None:
        """(key_hash, name, limit) or None if the key is unknown."""
        key_hash = hash_key(api_key)
        with self._lock:
            row = self._conn.execute("SELECT name, tpm_limit FROM tenants WHERE key_hash = ?", (key_hash,)).fetchone()
        if row is None:
            return None
        return key_hash, row[0], row[1] or self.default_limit

    # ------------------------------------------------------------------ core (sync; call via the async wrappers)
    def _window(self, key_hash: str, now: int, limit: int) -> tuple[WindowState, list[tuple[int, int]]]:
        cutoff = now - self.window_ms
        # Evict expired entries for this key as part of the same transaction.
        self._conn.execute("DELETE FROM token_usage WHERE key_hash = ? AND ts_ms <= ?", (key_hash, cutoff))
        rows = self._conn.execute(
            "SELECT ts_ms, tokens FROM token_usage WHERE key_hash = ? ORDER BY ts_ms, id", (key_hash,)
        ).fetchall()
        used = sum(t for _, t in rows)
        reset = (rows[0][0] + self.window_ms - now) if rows else 0
        return WindowState(limit=limit, used=used, reset_ms=max(0, reset)), rows

    def reserve_sync(self, key_hash: str, tokens: int, limit: int | None = None, request_id: str | None = None) -> Reservation | RateLimited:
        limit = limit or self.default_limit
        if tokens > limit:
            raise RequestTooLarge(tokens, limit)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                now = self.now_ms()
                state, rows = self._window(key_hash, now, limit)
                if state.used + tokens > limit:
                    self._conn.execute("COMMIT")  # keep the eviction
                    return RateLimited(self._retry_after(rows, state.used, tokens, limit, now), state)
                cur = self._conn.execute(
                    "INSERT INTO token_usage (key_hash, ts_ms, tokens, request_id) VALUES (?, ?, ?, ?)",
                    (key_hash, now, tokens, request_id),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        reset = state.reset_ms if rows else self.window_ms
        assert cur.lastrowid is not None
        return Reservation(cur.lastrowid, key_hash, tokens, WindowState(limit, state.used + tokens, reset))

    def _retry_after(self, rows: list[tuple[int, int]], used: int, tokens: int, limit: int, now: int) -> int:
        """Earliest moment enough old entries expire for ``tokens`` to fit: walk oldest-first."""
        need_to_free = used + tokens - limit
        freed = 0
        for ts, t in rows:
            freed += t
            if freed >= need_to_free:
                return max(1, ts + self.window_ms - now)  # entry is evicted once ts <= now - window
        return self.window_ms  # unreachable while tokens <= limit

    def reconcile_sync(self, reservation_id: int, actual_tokens: int, key_hash: str | None = None) -> None:
        """Set the reservation to the actual usage. Actual > reserved is recorded as-is (the tenant
        goes into "debt" and the next requests wait), so usage can't be under-counted.

        If the row is gone - a stream that outlived the window had it evicted - the usage is
        re-inserted at *now*: tokens consumed are never silently forgiven."""
        tokens = max(0, int(actual_tokens))
        with self._lock:
            cur = self._conn.execute("UPDATE token_usage SET tokens = ? WHERE id = ?", (tokens, reservation_id))
            if cur.rowcount == 0 and key_hash is not None and tokens > 0:
                self._conn.execute(
                    "INSERT INTO token_usage (key_hash, ts_ms, tokens, request_id) VALUES (?, ?, ?, ?)",
                    (key_hash, self.now_ms(), tokens, f"late:{reservation_id}"),
                )

    def release_sync(self, reservation_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM token_usage WHERE id = ?", (reservation_id,))

    def state_sync(self, key_hash: str, limit: int | None = None) -> WindowState:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                state, _ = self._window(key_hash, self.now_ms(), limit or self.default_limit)
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return state

    def prune_sync(self) -> int:
        """Delete every expired row (keys that went idle are otherwise never revisited)."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM token_usage WHERE ts_ms <= ?", (self.now_ms() - self.window_ms,))
        return cur.rowcount

    def row_count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ async API (never block the event loop)
    async def reserve(self, key_hash: str, tokens: int, limit: int | None = None, request_id: str | None = None):
        return await asyncio.to_thread(self.reserve_sync, key_hash, tokens, limit, request_id)

    async def reconcile(self, reservation_id: int, actual_tokens: int, key_hash: str | None = None) -> None:
        await asyncio.to_thread(self.reconcile_sync, reservation_id, actual_tokens, key_hash)

    async def release(self, reservation_id: int) -> None:
        await asyncio.to_thread(self.release_sync, reservation_id)

    async def state(self, key_hash: str, limit: int | None = None) -> WindowState:
        return await asyncio.to_thread(self.state_sync, key_hash, limit)

    async def prune(self) -> int:
        return await asyncio.to_thread(self.prune_sync)
