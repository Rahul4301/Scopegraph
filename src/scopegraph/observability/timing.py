"""Small monotonic timing helpers used by evaluation and service code."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager


@contextmanager
def elapsed_ms() -> Iterator[Callable[[], float]]:
    started = time.perf_counter()
    yield lambda: (time.perf_counter() - started) * 1000
