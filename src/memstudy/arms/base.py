"""Arm interface shared by A (full context), B (Mem0), and C (plain RAG)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from memstudy.metering import CostSink
from memstudy.schema import History, Item
from memstudy.tokens import count_tokens


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
    candidates: int = 0
    retrieval_seconds: float = 0.0
    retrieval_cost: CostSink = field(default_factory=CostSink)


def fill_to_budget(texts: list[str], budget: int) -> list[str]:
    """Whole retrieved items in rank order until the next one would exceed the token budget.

    Arms B, C and D all use this rule with the same budget, so they differ in what is retrieved
    and not in how much context the reader gets. Nothing is cut mid-item.
    """
    kept: list[str] = []
    used = 0
    for text in texts:
        cost = count_tokens(text) + 2  # list marker and newline
        if used + cost > budget:
            break
        kept.append(text)
        used += cost
    return kept


class Arm(Protocol):
    name: str

    def prepare(self, history: History) -> IngestStats | None:
        """Ingest a history once. Returns None when there is no ingestion step."""
        ...

    def context(self, item: Item, history: History) -> ArmContext:
        """Build the context for one question. Never truncates."""
        ...
