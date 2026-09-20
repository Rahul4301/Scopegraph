"""Stale-memory retrieval diagnostics."""

from collections.abc import Iterable
from typing import Any


def stale_memory_error_rate(retrieved: Iterable[Any]) -> float:
    items = list(retrieved)
    if not items:
        return 0.0
    stale = sum(
        (item.get("status") if isinstance(item, dict) else item.status) != "active"
        for item in items
    )
    return stale / len(items)
