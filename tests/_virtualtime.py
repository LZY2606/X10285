"""Deterministic virtual-time machinery for the PyrateLimiter contract suite.

Scope: tests only. Everything here drives the library through its *public*
entry points - ``Limiter(...)``, ``try_acquire`` / ``try_acquire_async`` and
the public ``AbstractClock`` / ``AbstractBucket`` extension points. No
internal result objects are fabricated.

Two virtual-time front ends share one millisecond clock:

* :class:`SyncVirtualClock` is a sync ``AbstractClock`` for bucket timestamps
  paired with a sync ``sleep`` that parks until the test driver advances time.
* :class:`AsyncVirtualClock` is an awaitable ``AbstractClock``; the
  :class:`DeterministicEventLoop` moves the *same* clock forward whenever the
  asyncio loop idles, so ``asyncio.sleep`` costs no wall-clock time.

The clock is the happens-before authority: a worker can observe an instant
only after the driver has advanced to it (barriers pin the *order* in which
workers arrive at that instant). OS thread scheduling never decides a case.
"""

from __future__ import annotations

import asyncio
import threading
from heapq import heappop
from typing import Any, Callable, List, Optional

from pyrate_limiter.clocks import AbstractClock


class ContractViolation(AssertionError):
    """Raised at the FIRST violated contract clause, carrying the execution
    trace so failures reproduce without re-running anything."""


class VirtualClock:
    """Shared virtual millisecond clock + sync parking lot.

    ``advance(delta_ms)`` is the only way time moves. Threads blocked in
    :meth:`sleep_ms` are released one at a time in FIFO arrival order whenever
    their deadline is reached, so the sync schedule is deterministic.
    """

    WATCHDOG_S = 10

    def __init__(self, start_ms: int = 1_000_000):
        self._cond = threading.Condition()
        self._now_ms = start_ms
        self._arrivals = 0
        self._parked: List[List[Any]] = []  # [deadline_ms, arrival_seq, token]
        self._wake_order: List[int] = []

    def now_ms(self) -> int:
        with self._cond:
            return self._now_ms

    def advance(self, delta_ms: int) -> int:
        """Move time forward exactly ``delta_ms``; release every sleeper whose
        deadline is met, one FIFO step at a time, letting released workers
        re-park before the next release. Returns the new virtual instant."""
        assert delta_ms >= 0, "virtual time cannot move backwards"
        with self._cond:
            self._now_ms += delta_ms
            while True:
                due = sorted(
                    (slot for slot in self._parked if slot[2] is None and slot[0] <= self._now_ms),
                    key=lambda slot: (slot[0], slot[1]),
                )
                if not due:
                    break
                slot = due[0]
                slot[2] = object()
                self._wake_order.append(slot[1])
                self._cond.notify_all()
                # Wait until the released worker has either returned from
                # sleep_ms or parked again at a later deadline, so the next
                # FIFO release sees its effects (re-put / re-wait) in order.
                settled = self._cond.wait_for(lambda: self._settled(slot), timeout=self.WATCHDOG_S)
                if not settled:
                    raise ContractViolation("virtual clock watchdog: released worker never settled")
            return self._now_ms

    def _settled(self, slot: List[Any]) -> bool:
        """A woken worker is settled once it no longer waits on this token."""
        return slot not in self._parked or slot[2] is None or slot[0] > self._now_ms

    def sleep_ms(self, duration_ms: int) -> None:
        """Block the calling worker thread until virtual time covers the sleep."""
        duration_ms = max(0, int(duration_ms))
        with self._cond:
            if duration_ms == 0:
                return
            seq = self._arrivals
            self._arrivals += 1
            slot = [self._now_ms + duration_ms, seq, None]
            self._parked.append(slot)
            self._cond.notify_all()  # tell the driver a worker parked
            while slot in self._parked and slot[2] is None:
                if not self._cond.wait_for(
                    lambda: slot not in self._parked or slot[2] is not None,
                    timeout=self.WATCHDOG_S,
                ):
                    raise ContractViolation("virtual clock watchdog: worker was never released")
            self._parked.remove(slot)
            self._cond.notify_all()

    def parked_deadlines(self) -> List[int]:
        with self._cond:
            return sorted(slot[0] for slot in self._parked if slot[2] is None)

    def parked_count(self) -> int:
        with self._cond:
            return sum(1 for slot in self._parked if slot[2] is None)

    def wait_until_parked(self, count: int = 1, timeout: float = 5.0) -> None:
        """Driver-side wait until ``count`` workers sleep virtually."""
        with self._cond:
            ok = self._cond.wait_for(lambda: self.parked_count >= count, timeout=timeout)
        if not ok:
            raise ContractViolation(
                f"expected {count} parked worker(s), observed {self.parked_count()} after {timeout}s"
            )

    def wake_order(self) -> List[int]:
        with self._cond:
            return list(self._wake_order)


class SyncVirtualClock(AbstractClock):
    """Sync ``AbstractClock`` + sync sleep backend over a :class:`VirtualClock`."""

    def __init__(self, vclock: Optional[VirtualClock] = None):
        self.vclock = vclock or VirtualClock()

    def now(self) -> int:
        return self.vclock.now_ms()

    def monotonic(self) -> float:
        return self.vclock.now_ms() / 1000.0

    def sleep(self, seconds: float) -> None:
        self.vclock.sleep_ms(int(round(max(0.0, seconds) * 1000)))


class AsyncVirtualClock(AbstractClock):
    """Awaitable ``AbstractClock`` over the same :class:`VirtualClock`."""

    async def now(self) -> int:
        return self.vclock.now_ms()

    def __init__(self, vclock: VirtualClock):
        self.vclock = vclock


class _LoopStalled(RuntimeError):
    """The deterministic loop ran out of both ready callbacks and timers."""


class DeterministicEventLoop(asyncio.SelectorEventLoop):
    """Selector loop whose clock and idle progress are virtual.

    Ready callbacks always run first; when none are ready the loop jumps the
    shared virtual clock to the earliest scheduled ``call_later`` deadline and
    fires it. Equal-deadline timers fire in heap/FIFO order, matching the real
    loop - so schedules are stable and independent of OS timing.
    """

    def __init__(self, vclock: VirtualClock):
        super().__init__()
        self._vclock = vclock

    def time(self) -> float:
        return self._vclock.now_ms() / 1000.0

    def _run_once(self) -> None:
        while self._ready:
            handle = self._ready.popleft()
            if handle._cancelled:
                continue
            handle._run()

        sched = self._scheduled
        while sched and sched[0]._cancelled:
            self._timer_cancelled_count -= 1
            heappop(sched)

        if not sched:
            raise _LoopStalled()

        handle = heappop(sched)
        if handle._cancelled:
            self._timer_cancelled_count -= 1
            return

        delta_ms = max(0, int(round(handle._when * 1000 - self._vclock.now_ms())))
        self._vclock.advance(delta_ms)
        if not handle._cancelled:
            handle._run()


def new_loop(vclock: VirtualClock) -> DeterministicEventLoop:
    loop = DeterministicEventLoop(vclock)
    asyncio.set_event_loop(loop)
    return loop


def close_loop(loop: DeterministicEventLoop) -> None:
    try:
        loop.close()
    finally:
        asyncio.set_event_loop(None)


def pump(loop: DeterministicEventLoop, max_ready: int = 100_000) -> int:
    """Run every currently-ready callback (and callbacks they enqueue) without
    advancing virtual time. Returns how many handles ran."""
    ran = 0
    while loop._ready and ran < max_ready:
        handle = loop._ready.popleft()
        if handle._cancelled:
            continue
        handle._run()
        ran += 1
    return ran


def next_timer_ms(loop: DeterministicEventLoop) -> Optional[int]:
    sched = loop._scheduled
    while sched and sched[0]._cancelled:
        loop._timer_cancelled_count -= 1
        heappop(sched)
    if not sched:
        return None
    return max(0, int(round(sched[0]._when * 1000 - loop._vclock.now_ms())))


def run_deterministically(vclock: VirtualClock, factory: Callable[[], Any], max_steps: int = 100_000):
    """Drive ``factory()`` (a coroutine function or coroutine) to completion on
    a fresh deterministic loop bound to ``vclock``."""
    loop = new_loop(vclock)
    try:
        coro = factory() if callable(factory) else factory
        task = loop.create_task(coro)
        steps = 0
        while not task.done():
            loop._run_once()
            steps += 1
            if steps > max_steps:
                raise ContractViolation("deterministic loop exceeded step budget")
        return task.result()
    finally:
        close_loop(loop)


class AsyncBarrier:
    """Coroutine barrier on the deterministic loop.

    Unlike ``asyncio.Barrier`` it schedules no timer: parties park on futures
    (no ready callbacks), so the driver controls exactly when all parked
    parties are released together with ``release_all``.
    """

    def __init__(self, parties: int):
        self.parties = parties
        self._arrived = 0
        self._waiters: List[asyncio.Future] = []

    async def arrive(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        loop = loop or asyncio.get_running_loop()
        future = loop.create_future()
        self._waiters.append(future)
        self._arrived += 1
        await future

    @property
    def arrived(self) -> int:
        return self._arrived

    def release_all(self, loop: asyncio.AbstractEventLoop) -> None:
        assert self._arrived >= self.parties, "barrier released before all parties arrived"
        waiters, self._waiters = self._waiters, []
        self._arrived = 0
        for waiter in waiters:
            if not waiter.done():
                loop.call_soon(waiter.set_result, None)
