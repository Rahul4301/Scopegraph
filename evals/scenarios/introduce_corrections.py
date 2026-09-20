"""Structured correction plans shared by correction experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CorrectionPlan:
    condition: str
    actor: str = "eval"
    reason: str = "correction-persistence experiment"


def correction_conditions() -> tuple[CorrectionPlan, ...]:
    return tuple(CorrectionPlan(condition=name) for name in ("none", "conversational", "graph"))
