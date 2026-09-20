"""Small, dependency-free paired bootstrap helpers."""

import random
from collections.abc import Sequence


def paired_difference(left: Sequence[float], right: Sequence[float]) -> list[float]:
    if len(left) != len(right):
        raise ValueError("paired samples must have equal length")
    return [float(a) - float(b) for a, b in zip(left, right, strict=True)]


def bootstrap_mean_difference(left: Sequence[float], right: Sequence[float], *, seed: int = 42,
                             samples: int = 2000) -> dict[str, float]:
    differences = paired_difference(left, right)
    if not differences:
        return {"mean": 0.0, "lower_95": 0.0, "upper_95": 0.0}
    rng = random.Random(seed)
    means = [sum(rng.choice(differences) for _ in differences) / len(differences)
             for _ in range(samples)]
    means.sort()
    return {
        "mean": sum(differences) / len(differences),
        "lower_95": means[int(0.025 * (len(means) - 1))],
        "upper_95": means[int(0.975 * (len(means) - 1))],
    }
