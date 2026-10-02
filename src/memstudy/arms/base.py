"""Arm interface shared by A (full context), B (Mem0), and C (plain RAG)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from memstudy.metering import CostSink
from memstudy.schema import History, Item


@dataclass
class IngestStats:
    """Cost and time of turning one history into the arm's store (zero for arm A)."""

    seconds: float
    cost: CostSink
    stored_units: int


@dataclass
class ArmContext:
    text: str
    context_tokens: int
    cache_prefix: bool
    retrieved: list[str] = field(default_factory=list)
    retrieval_seconds: float = 0.0
    retrieval_cost: CostSink = field(default_factory=CostSink)


class Arm(Protocol):
    name: str

    def prepare(self, history: History) -> IngestStats | None:
        """Ingest a history once. Returns None when there is no ingestion step."""
        ...

    def context(self, item: Item, history: History) -> ArmContext:
        """Build the context for one question. Never truncates."""
        ...
