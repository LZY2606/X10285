"""Test-only doubles for the virtual-time concurrency & rollback contract suite
(虚拟时间并发与回滚契约套件).

Visibility & production-logic sharing (per the suite's constraints):

- Everything here lives under ``tests/`` and is never shipped; the production
  package is untouched, so old entry points and old data keep working.
- ``VirtualClock`` implements the public ``AbstractClock`` interface - the same
  extension point production clocks (``MonotonicClock``, ``SQLiteClock``) use.
- ``FlakyBucket`` / ``GateBucket`` subclass the production ``InMemoryBucket``
  and reuse its ``put``/``waiting``/``leak`` verbatim; they only inject a
  failure or a happens-before signal *around* the same production call. No
  rate-limit algorithm is re-implemented here (no parallel implementation).
- ``KeyedVirtualFactory`` routes items through the public ``BucketFactory``
  contract, mirroring ``SingleBucketFactory`` but one bucket per key.
- Every acquisition in the suite enters through ``Limiter.try_acquire`` /
  ``Limiter.try_acquire_async``; buckets are only *observed* through their
  public surface (``count``, ``peek``, ``waiting``, ``failing_rate``, ``leak``,
  ``flush``).
"""
import threading
from typing import Dict, List, Optional, Tuple

from pyrate_limiter import (
    AbstractBucket,
    AbstractClock,
    BucketFactory,
    InMemoryBucket,
    Limiter,
    Rate,
    RateItem,
    SingleBucketFactory,
)


class VirtualClock(AbstractClock):
    """Deterministic clock: time moves only when the test says so.

    Implements the same public ``AbstractClock`` contract as the production
    clocks, so buckets use it through ``bucket._clock`` exactly like
    ``SQLiteClock`` is swapped in by the existing test-suite.
    """

    def __init__(self, start: int = 1_700_000_000_000):
        self.t = start

    def now(self) -> int:
        return self.t

    def advance(self, ms: int) -> int:
        self.t += ms
        return self.t


class ContractLog:
    """Ordered record of caller-visible results and internal invariants.

    Records are appended in execution order (stable ordering for diagnostics).
    The first ``check`` violation raises an ``AssertionError`` that names the
    failing step and dumps the whole ordered log, so a failure always shows
    *where* the contract broke and the context that led there - never a bare
    true/false.
    """

    def __init__(self, title: str):
        self.title = title
        self.records: List[Tuple[str, str]] = []

    def record(self, step: str, **fields) -> None:
        detail = " ".join(f"{key}={value}" for key, value in fields.items())
        self.records.append((step, detail))

    def check(self, condition: bool, step: str, **context) -> None:
        self.record(step, verdict="ok" if condition else "VIOLATION", **context)
        if not condition:
            raise AssertionError(f"contract violated at step {step!r} [{self.title}]\n{self.dump()}")

    def vector(self) -> Tuple[Tuple[str, str], ...]:
        """The ordered records, for equality comparison across runs."""
        return tuple(self.records)

    def dump(self) -> str:
        lines = [f"contract log [{self.title}] (last entry is the first violation):"]
        lines += [f"  [{idx:02d}] {step}: {detail}" for idx, (step, detail) in enumerate(self.records)]
        return "\n".join(lines)


def snapshot(bucket: AbstractBucket, clock: VirtualClock, weight: int = 1) -> Dict[str, object]:
    """Internal invariants proving limiter/rollback semantics, via public reads."""
    failing = bucket.failing_rate
    return {
        "count": bucket.count(),
        "failing": None if failing is None else failing.interval,
        "wait": bucket.waiting(RateItem("probe", clock.now(), weight=weight)),
    }


class FlakyBucket(InMemoryBucket):
    """InMemoryBucket that injects a recoverable failure on demand.

    ``after_write=False`` raises *before* the write (transient backend outage:
    nothing is committed); ``after_write=True`` lets the production ``put``
    commit and then raises (the write landed but the caller sees a failure).
    Either way the original exception propagates through ``Limiter`` unchanged,
    which is the existing error model (``Limiter`` only translates
    ``TimeoutError`` into ``False``).
    """

    def __init__(self, rates: List[Rate]):
        super().__init__(rates)
        self._armed = 0
        self._after_write = False
        self.injected = 0

    def arm(self, times: int = 1, after_write: bool = False) -> None:
        self._armed = times
        self._after_write = after_write

    def put(self, item: RateItem) -> bool:
        if self._armed > 0 and not self._after_write:
            self._armed -= 1
            self.injected += 1
            raise ConnectionError("injected transient backend failure (before write)")

        result = super().put(item)

        if self._armed > 0 and self._after_write:
            self._armed -= 1
            self.injected += 1
            raise ConnectionError("injected transient backend failure (after write)")

        return result


class GateBucket(InMemoryBucket):
    """InMemoryBucket that signals when ``waiting()`` is consulted.

    Lets an async test establish happens-before without sleeping real time:
    the gate is set exactly when a denied acquirer has computed its delay and
    is about to park in the (virtual-time-derived) delay sleep.
    """

    def __init__(self, rates: List[Rate]):
        super().__init__(rates)
        self.waiting_gate = None

    def waiting(self, item: RateItem):
        if self.waiting_gate is not None:
            self.waiting_gate.set()
        return super().waiting(item)


class KeyedVirtualFactory(BucketFactory):
    """One InMemoryBucket per key, all sharing one VirtualClock.

    No background leak is scheduled: the suite advances the clock and calls
    ``leak()`` explicitly, keeping scheduling deterministic.
    """

    def __init__(self, rates: List[Rate], clock: VirtualClock):
        self._rates = list(rates)
        self._clock = clock
        self._buckets: Dict[str, InMemoryBucket] = {}

    def wrap_item(self, name: str, weight: int = 1) -> RateItem:
        return RateItem(name, self._clock.now(), weight=weight)

    def get(self, item: RateItem) -> InMemoryBucket:
        bucket = self._buckets.get(item.name)
        if bucket is None:
            bucket = InMemoryBucket(list(self._rates))
            bucket._clock = self._clock
            self._buckets[item.name] = bucket
        return bucket


class AsyncBarrier:
    """Deterministic asyncio barrier (py3.10-compatible).

    Every party parks until the last one arrives; the release is the
    happens-before edge for the concurrent attempt that follows.
    """

    def __init__(self, parties: int):
        import asyncio

        self._parties = parties
        self._arrived = 0
        self._gate = asyncio.Event()

    async def wait(self) -> None:
        self._arrived += 1
        if self._arrived >= self._parties:
            self._gate.set()
        await self._gate.wait()


def make_limiter(rates: Optional[List[Rate]] = None, clock: Optional[VirtualClock] = None, bucket: Optional[AbstractBucket] = None):
    """Build a Limiter on a virtual clock with no background leaker.

    ``schedule_leak=False`` is the documented public option for driving
    ``leak()`` manually - that is what keeps the suite's scheduling
    deterministic and free of real-time sleeps.
    """
    clock = clock or VirtualClock()
    if bucket is None:
        assert rates is not None
        bucket = InMemoryBucket(list(rates))
    bucket._clock = clock
    limiter = Limiter(SingleBucketFactory(bucket, schedule_leak=False))
    return limiter, bucket


def sync_entry(limiter: Limiter):
    """Adapt the sync public entry to the shared script's async callable shape."""

    async def acquire(name: str, **kwargs) -> bool:
        return limiter.try_acquire(name, **kwargs)

    return acquire


def async_entry(limiter: Limiter):
    """Adapt the async public entry to the shared script's async callable shape."""

    async def acquire(name: str, **kwargs) -> bool:
        return await limiter.try_acquire_async(name, **kwargs)

    return acquire


async def window_delay_script(acquire, bucket: AbstractBucket, clock: VirtualClock, log: ContractLog) -> None:
    """Shared sync/async contract: window endpoints, delay and leak scheduling.

    Assumes ``Rate(3, 1000)``. Every step enters through the limiter acquire
    callable; the bucket is only observed through its public surface. No real
    time is slept: the virtual clock is the only scheduler.
    """
    t0 = clock.now()

    for i in range(1, 4):
        ok = await acquire("k", blocking=False)
        log.check(ok is True, f"admit-{i}-of-3", **snapshot(bucket, clock))

    ok = await acquire("k", blocking=False)
    log.check(ok is False, "deny-at-limit", **snapshot(bucket, clock))
    log.check(bucket.failing_rate is bucket.rates[0], "failing-rate-recorded")
    wait = bucket.waiting(RateItem("k", clock.now()))
    log.check(wait == 1001, "delay-equals-window-plus-one", wait=wait)

    clock.advance(999)
    ok = await acquire("k", blocking=False)
    log.check(ok is False, "boundary-minus-one-still-denied", **snapshot(bucket, clock))
    wait = bucket.waiting(RateItem("k", clock.now()))
    log.check(wait == 2, "boundary-minus-one-wait", wait=wait)

    clock.advance(1)
    ok = await acquire("k", blocking=False)
    log.check(ok is False, "boundary-inclusive-still-denied", **snapshot(bucket, clock))
    wait = bucket.waiting(RateItem("k", clock.now()))
    log.check(wait == 1, "boundary-wait-is-one-ms", wait=wait)

    clock.advance(1)
    ok = await acquire("k", blocking=False)
    log.check(ok is True, "admit-after-full-window", **snapshot(bucket, clock))
    log.check(bucket.count() == 4, "count-tracks-admissions", count=bucket.count())

    newest, oldest = bucket.peek(0), bucket.peek(3)
    log.check(
        newest is not None and oldest is not None and newest.timestamp == t0 + 1001 and oldest.timestamp == t0,
        "peek-order-latest-to-earliest",
    )

    leaked = bucket.leak(clock.now())
    log.check(leaked == 3 and bucket.count() == 1, "manual-leak-expires-out-of-window", leaked=leaked)


def run_fairness_round(limiter: Limiter, bucket: AbstractBucket, capacity: int, order: List[int], log: ContractLog, label: str) -> set:
    """One barrier-synchronized concurrent round at a frozen virtual instant.

    The barrier release is the happens-before edge: every worker attempts its
    first acquire at the same virtual time. A ``False`` paired with
    ``count < capacity`` can only be lock contention on the non-blocking
    limiter lock (the existing ``TimeoutError -> False`` translation), never a
    capacity verdict, so the worker retries; a denial is decisive only once the
    window is provably full. Assertions are order-independent: they check how
    many and which invariants hold, never which thread won.
    """
    barrier = threading.Barrier(len(order))
    results: Dict[int, bool] = {}
    errors: Dict[int, str] = {}

    def worker(i: int) -> None:
        barrier.wait()
        try:
            for _ in range(500):
                if limiter.try_acquire(f"w{i}", blocking=False):
                    results[i] = True
                    return
                if bucket.count() >= capacity:
                    results[i] = False
                    return
            errors[i] = "retry budget exhausted"
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            errors[i] = repr(exc)

    threads = [threading.Thread(target=worker, args=(i,), name=f"fairness-{label}-{i}", daemon=True) for i in order]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    stuck = [thread.name for thread in threads if thread.is_alive()]
    log.check(not stuck, f"{label}-no-stuck-thread", stuck=stuck)
    log.check(not errors, f"{label}-no-exceptions", errors=errors)
    admitted = sorted(i for i, ok in results.items() if ok)
    denied = sorted(i for i, ok in results.items() if not ok)
    log.check(len(admitted) == capacity, f"{label}-admitted-equals-capacity", admitted=admitted)
    log.check(len(denied) == len(order) - capacity, f"{label}-denied-equals-rest", denied=denied)
    log.check(bucket.count() == capacity, f"{label}-count-equals-capacity", count=bucket.count())
    names = {bucket.peek(i).name for i in range(bucket.count())}
    log.check(len(names) == capacity, f"{label}-stored-names-distinct", names=sorted(names))
    return set(admitted)
