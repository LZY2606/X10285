"""Virtual-time harness for the limiter concurrency & rollback contract suite.

One virtual timeline (milliseconds) drives both the bucket clock and the
limiter's sleep/deadline primitives, so contract tests never sleep real time
and every schedule is deterministic:

- ``VirtualTime`` is an ``AbstractClock``; attaching it to a bucket
  (``bucket._clock = vtime``) follows the same convention as
  ``tests/test_gcra.py``'s ``FrozenClock``.
- The ``virtual_time`` fixture (in ``tests/test_virtual_time_contract.py``)
  monkeypatches ``pyrate_limiter.limiter.sleep`` / ``...monotonic`` and
  ``asyncio.sleep`` with the virtual equivalents. No limiter or bucket object
  is faked: every scenario still enters through the public ``Limiter``
  (``try_acquire`` / ``try_acquire_async`` / decorator) and the real
  ``InMemoryBucket``; only *time* is virtualized.

Happens-before is expressed with ``threading.Barrier`` (threads) and explicit
``VirtualTime.advance`` calls, never with real-time sleeps or assumed thread
interleavings.

Note on naming: v3's ``max_delay`` is v4's ``timeout`` parameter of
``try_acquire`` / ``try_acquire_async`` (maximum time to wait before giving
up). The contract tests use the v4 public name.
"""

from __future__ import annotations

import asyncio
import threading
from typing import List, Optional, Sequence, Tuple

from pyrate_limiter import AbstractClock, InMemoryBucket, Limiter, Rate, SingleBucketFactory
from pyrate_limiter.abstracts.algorithm import SlidingWindowLog

# Captured before any test patches asyncio.sleep; the virtual async sleep still
# yields to the event loop (one real zero-second sleep) so tasks interleave and
# cancellation lands deterministically, without consuming real time.
_REAL_ASYNC_SLEEP = asyncio.sleep


async def yield_to_loop(times: int = 1) -> None:
    """Let other tasks run without touching the virtual timeline.

    Uses the real zero-second sleep captured at import time, so it stays a
    pure scheduling yield even while ``asyncio.sleep`` is patched.
    """
    for _ in range(times):
        await _REAL_ASYNC_SLEEP(0)


class ContractBreach(AssertionError):
    """First contract violation, raised with the virtual-time context attached."""


def require(condition: bool, contract: str, **context) -> None:
    """Assert a contract clause, keeping the first violation location and context."""
    if not condition:
        details = ", ".join(f"{key}={value!r}" for key, value in context.items())
        raise ContractBreach(f"{contract}: {details}")


class VirtualTime(AbstractClock):
    """A manually advanced clock shared by the bucket and the limiter.

    ``sleeps`` records every virtual sleep the limiter scheduled, in order, so
    tests can assert the exact deterministic schedule (delay / max_delay)
    instead of measuring wall time.
    """

    def __init__(self, start_ms: int = 1_000_000):
        self.now_ms = start_ms
        self.sleeps: List[int] = []
        self._lock = threading.Lock()

    # -- AbstractClock interface (bucket side)
    def now(self) -> int:
        return self.now_ms

    def advance(self, ms: int) -> int:
        with self._lock:
            self.now_ms += ms
            return self.now_ms

    # -- limiter primitives (patched in by the virtual_time fixture)
    def monotonic(self) -> float:
        """Replacement for ``time.monotonic`` (seconds), for limiter deadlines."""
        return self.now_ms / 1000

    def sleep(self, secs: float) -> None:
        """Sync replacement for ``time.sleep``: jump, never wait."""
        ms = round(secs * 1000)
        with self._lock:
            self.sleeps.append(ms)
            self.now_ms += ms

    async def async_sleep(self, secs: float) -> None:
        """Async replacement for ``asyncio.sleep``.

        The sleep is recorded first (so a cancelled waiter still shows up in
        the schedule), then the coroutine yields once; virtual time only
        advances if the waiter is not cancelled at that point - a cancelled
        wait therefore leaves no trace on the shared timeline either.
        """
        ms = round(secs * 1000)
        with self._lock:
            self.sleeps.append(ms)
        await _REAL_ASYNC_SLEEP(0)
        with self._lock:
            self.now_ms += ms


def make_limiter(
    vtime: VirtualTime,
    rates: Sequence[Rate],
    *,
    buffer_ms: int = 0,
    bucket: Optional[InMemoryBucket] = None,
) -> Tuple[Limiter, InMemoryBucket]:
    """Build a Limiter over a real InMemoryBucket on the virtual timeline.

    ``schedule_leak=False`` keeps the background Leaker thread out of the
    scenario: leaking is triggered explicitly via ``bucket.leak(vtime.now_ms)``
    so the happens-before relation stays test-controlled.
    """
    if bucket is None:
        bucket = InMemoryBucket(list(rates))
    bucket._clock = vtime
    limiter = Limiter(SingleBucketFactory(bucket, schedule_leak=False), buffer_ms=buffer_ms)
    return limiter, bucket


def snapshot(bucket: InMemoryBucket) -> List[Tuple[str, int, int]]:
    """Stable-ordered (latest-to-earliest) view of the bucket's stored items."""
    return [(bucket.peek(idx).name, bucket.peek(idx).weight, bucket.peek(idx).timestamp) for idx in range(bucket.count())]


class TransientStoreError(IOError):
    """Recoverable backend failure injected by ``FlakyBucket``."""


class FlakyBucket(InMemoryBucket):
    """InMemoryBucket that raises a recoverable error on chosen ``put`` calls.

    ``fail_after_write=False`` models a store that rejects the write (nothing
    persisted); ``fail_after_write=True`` models a store that persists and
    then fails to acknowledge - the write must still be all-or-nothing.
    """

    def __init__(self, rates, *, fail_on_calls, fail_after_write: bool = False):
        super().__init__(list(rates))
        self.fail_on_calls = frozenset(fail_on_calls)
        self.fail_after_write = fail_after_write
        self.put_calls = 0
        self.failures: List[str] = []

    def put(self, item):
        self.put_calls += 1
        if self.put_calls in self.fail_on_calls:
            marker = f"put#{self.put_calls}(item={item.name},weight={item.weight},after_write={self.fail_after_write})"
            if self.fail_after_write:
                super().put(item)
            self.failures.append(marker)
            raise TransientStoreError(f"recoverable store failure at {marker}")
        return super().put(item)


class TardyWindowMutant(SlidingWindowLog):
    """Deliberately wrong policy: the window lower bound is 1ms too generous,
    so an entry is still counted 1ms after it should have expired.

    Used to prove the window-endpoint contract actually detects a broken
    implementation (mutation testing).
    """

    def window_start(self, rate, now: int) -> int:
        return now - rate.interval - 1
