"""Cross-Scope Contamination Rate (CSCR)."""

from collections.abc import Iterable
from typing import Any


def cross_scope_contamination(retrieved: Iterable[Any], valid_scope_ids: Iterable[str]) -> float:
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
    return float(predicted_scope_id in set(gold_scope_ids))
