"""Token usage summaries."""

from collections.abc import Iterable


def token_summary(values: Iterable[int]) -> dict[str, float]:
    tokens = [int(value) for value in values]
    return {
        "count": float(len(tokens)),
        "total": float(sum(tokens)),
        "mean": float(sum(tokens) / len(tokens)) if tokens else 0.0,
        "max": float(max(tokens)) if tokens else 0.0,
    }
