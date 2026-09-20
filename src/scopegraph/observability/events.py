"""Structured, JSON-safe experiment event helpers."""

from datetime import UTC, datetime
from typing import Any


def event(name: str, **fields: Any) -> dict[str, Any]:
    return {"name": name, "timestamp": datetime.now(UTC).isoformat(), **fields}
