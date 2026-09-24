import pytest
from .conftest import DEFAULT_RATES
from pyrate_limiter import Limiter
from pyrate_limiter import SingleBucketFactory

@pytest.mark.asyncio
async def test_multiple_bucket_closes(
    create_bucket,
):
    # Makes sure no exceptions even if close is called multiple times
    
    with await create_bucket(DEFAULT_RATES) as bucket:
        bucket.close()
    bucket.close()


@pytest.mark.asyncio
async def test_limiter_close_closes_its_buckets(
    create_bucket,
):
    # Makes sure closing the limiter closes the buckets it owns

    bucket = await create_bucket(DEFAULT_RATES)

    calls = []
    real_close = bucket.close

    def spy():
        calls.append(bucket)
        real_close()

    bucket.close = spy

    limiter = Limiter(bucket)
    limiter.close()

    assert calls, "Limiter.close() must close the buckets it owns"


@pytest.mark.asyncio
async def test_limiter_close_closes_bucket_without_scheduled_leak(
    create_bucket,
):
    # Makes sure the limiter closes its bucket even when no leak was scheduled

    bucket = await create_bucket(DEFAULT_RATES)

    calls = []
    real_close = bucket.close

    def spy():
        calls.append(bucket)
        real_close()

    bucket.close = spy

    limiter = Limiter(SingleBucketFactory(bucket, schedule_leak=False))
    limiter.close()

    assert calls, "Limiter.close() must close its bucket even without a leaker"


@pytest.mark.asyncio
async def test_multiple_bucket_closes_limiter(
    create_bucket,
):
    # Makes sure no exceptions even if close is called multiple times
    
    with await create_bucket(DEFAULT_RATES) as bucket:
        with Limiter(bucket) as limiter:
            limiter.close()
        bucket.close()
    bucket.close()
