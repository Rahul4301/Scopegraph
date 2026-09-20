"""Latency summaries for reproducible reports."""

from collections.abc import Iterable
from statistics import mean, pstdev


def latency_summary(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "stddev": 0.0}

    def percentile(percent: float) -> float:
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * percent)))
        return ordered[index]

    return {
        "count": float(len(ordered)), "mean": mean(ordered), "p50": percentile(0.5),
        "p95": percentile(0.95), "stddev": pstdev(ordered) if len(ordered) > 1 else 0.0,
    }
