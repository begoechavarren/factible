import logging
import time
from contextlib import contextmanager
from typing import Generator

_logger = logging.getLogger(__name__)


@contextmanager
def timer(name: str) -> Generator[None, None, None]:
    """Context manager to time a block of code."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    _logger.info(f"⏱️  {name}: {elapsed:.2f}s")
