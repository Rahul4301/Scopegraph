import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from scopegraph.models.memory import Memory


@dataclass(frozen=True)
class RankingWeights:
    semantic: float = 0.40
    scope: float = 0.25
    temporal: float = 0.15
    confidence: float = 0.08
    graph: float = 0.07
    recency: float = 0.05

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RankingWeights":
        weights = config.get("weights", {})
        defaults = cls()
        values = {
            field: float(weights.get(field, getattr(defaults, field)))
            for field in cls.__dataclass_fields__
        }
        return cls(**values)

    def validate(self) -> None:
        values = (
            self.semantic,
            self.scope,
            self.temporal,
            self.confidence,
            self.graph,
            self.recency,
        )
        if any(value < 0 for value in values):
            raise ValueError("Ranking weights cannot be negative")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
            raise ValueError("Ranking weights must sum to 1.0")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions must match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    raw = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return max(0.0, min(1.0, raw))


def recency_score(memory: Memory, *, now: datetime, half_life_days: float = 180.0) -> float:
    age_seconds = max(0.0, (now - memory.updated_at).total_seconds())
    age_days = age_seconds / 86_400
    return 2 ** (-age_days / half_life_days)


def final_score(
    *,
    semantic: float,
    scope: float,
    temporal: float,
    confidence: float,
    graph: float,
    recency: float,
    weights: RankingWeights,
) -> float:
    weights.validate()
    values = (semantic, scope, temporal, confidence, graph, recency)
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("Ranking components must be normalized to [0, 1]")
    return (
        weights.semantic * semantic
        + weights.scope * scope
        + weights.temporal * temporal
        + weights.confidence * confidence
        + weights.graph * graph
        + weights.recency * recency
    )
