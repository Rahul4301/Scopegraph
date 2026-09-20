"""Logical storage accounting for reproducible evaluation records."""

from typing import Any


def logical_bytes(stats: dict[str, Any], *, serialized_memory_bytes: int = 0) -> int:
    return (
        int(stats.get("memory_count", 0)) * 256
        + int(stats.get("relationship_count", 0)) * 96
        + serialized_memory_bytes
    )
