"""Rate limiter factory using LangChain's built-in InMemoryRateLimiter."""

from langchain_core.rate_limiters import InMemoryRateLimiter


def create_rate_limiter(rpm: int) -> InMemoryRateLimiter:
    """Create a LangChain-compatible rate limiter from requests-per-minute.

    Args:
        rpm: Maximum requests per minute.

    Returns:
        An InMemoryRateLimiter that can be passed directly to any
        LangChain BaseChatModel via the ``rate_limiter`` constructor kwarg.
    """
    return InMemoryRateLimiter(
        requests_per_second=rpm / 60.0,
        check_every_n_seconds=0.1,
        max_bucket_size=3,
    )
