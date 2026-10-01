"""Cross-Scope Contamination Rate (CSCR)."""

from collections.abc import Iterable
from typing import Any


def cross_scope_contamination(retrieved: Iterable[Any], valid_scope_ids: Iterable[str]) -> float:
    """Share of retrieved items stored outside the valid scopes."""
    items = list(retrieved)
    if not items:
        return 0.0
    valid = set(valid_scope_ids)
    wrong = sum(
        (item.get("scope_id") if isinstance(item, dict) else item.scope_id) not in valid
        for item in items
    )
    return wrong / len(items)


def scope_classification_accuracy(predicted_scope_id: str | None,
                                  gold_scope_ids: Iterable[str]) -> float:
    """1.0 if the predicted scope id is a gold scope, else 0.0."""
    return float(predicted_scope_id in set(gold_scope_ids))
