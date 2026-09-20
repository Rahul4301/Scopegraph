"""Trace collection helpers for retrieval instrumentation."""

from typing import Any


class TraceCollector:
    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []

    def add(self, **step: Any) -> None:
        self.steps.append(step)

    def as_dicts(self) -> list[dict[str, Any]]:
        return list(self.steps)
