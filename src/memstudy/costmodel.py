"""Closed-form cost per history for each arm, with queries per history (Q) as a variable.

All inputs are measured token counts and the prices in configs/prices.yaml. Used for the pilot
projection (Phase 2) and the cost crossover analysis (Phase 4).
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from memstudy.budget import ModelPrice


@dataclass(frozen=True)
class QueryShape:
    """Tokens of the non-history part of one query."""

    question_tokens: int
    answer_tokens: int


def _rates(price: ModelPrice, prompt_tokens: int) -> tuple[float, float, float, float]:
    m = price.multipliers(prompt_tokens)
    return (
        price.input * m["input"] / 1e6,
        price.cached_input * m["cached_input"] / 1e6,
        price.cache_write * m["cache_write"] / 1e6,
        price.output * m["output"] / 1e6,
    )


def full_context_cost(
    price: ModelPrice, history_tokens: int, queries: int, shape: QueryShape, cached: bool
) -> float:
    """Arm A cost for one history. With caching: one write of the history, then reads, assuming
    the cache stays warm between queries (30 minute lifetime refreshed on every reuse)."""
    prompt = history_tokens + shape.question_tokens
    r_in, r_cached, r_write, r_out = _rates(price, prompt)
    per_query_tail = shape.question_tokens * r_in + shape.answer_tokens * r_out
    if not cached:
        return queries * (history_tokens * r_in + per_query_tail)
    first = history_tokens * r_write + per_query_tail
    later = history_tokens * r_cached + per_query_tail
    return first + (queries - 1) * later if queries > 0 else 0.0


def memory_cost(
    price: ModelPrice,
    ingest_usd: float,
    queries: int,
    context_tokens: int,
    shape: QueryShape,
    retrieval_usd_per_query: float = 0.0,
) -> float:
    """Arm B, C or D cost for one history: ingestion once, then retrieval plus an uncached read."""
    prompt = context_tokens + shape.question_tokens
    r_in, _, _, r_out = _rates(price, prompt)
    per_query = prompt * r_in + shape.answer_tokens * r_out + retrieval_usd_per_query
    return ingest_usd + queries * per_query


def crossover_queries(
    price: ModelPrice,
    history_tokens: int,
    ingest_usd: float,
    context_tokens: int,
    shape: QueryShape,
    cached: bool,
    retrieval_usd_per_query: float = 0.0,
) -> float | None:
    """Smallest Q at which the memory arm is cheaper than full context, or None if the memory
    arm is never cheaper per query (no crossover exists at this history length)."""
    a1 = full_context_cost(price, history_tokens, 1, shape, cached)
    a2 = full_context_cost(price, history_tokens, 2, shape, cached)
    slope_a = a2 - a1
    intercept_a = a1 - slope_a
    b1 = memory_cost(price, ingest_usd, 1, context_tokens, shape, retrieval_usd_per_query)
    b2 = memory_cost(price, ingest_usd, 2, context_tokens, shape, retrieval_usd_per_query)
    slope_b = b2 - b1
    intercept_b = b1 - slope_b
    if slope_a <= slope_b:
        return None
    return max(0.0, (intercept_b - intercept_a) / (slope_a - slope_b))


# --- Query spacing -------------------------------------------------------------------------
# A cached prefix stays warm for the cache lifetime T after its last use (refreshed on reuse), so
# whether a query reads or rewrites the history depends on the gap since the previous query.

_TTL = re.compile(r"^(\d+(?:\.\d+)?)\s*(s|m|h)$")


def ttl_seconds(ttl: str) -> float:
    """Cache lifetime from the study config ("30m", "1h", "90s") in seconds."""
    m = _TTL.match(ttl.strip())
    if not m:
        raise ValueError(f"cannot read cache ttl {ttl!r}; use forms like 30m, 1h, 90s")
    return float(m.group(1)) * {"s": 1, "m": 60, "h": 3600}[m.group(2)]


def fixed_schedule(queries: int, gap: float) -> list[float]:
    """Arrival times of queries spaced evenly, gap seconds apart."""
    return [i * gap for i in range(queries)]


def burst_schedule(queries: int, burst_size: int, intra_gap: float, inter_gap: float) -> list[float]:
    """Bursts of burst_size queries intra_gap apart, bursts inter_gap apart (idle time between)."""
    times: list[float] = []
    t = 0.0
    for i in range(queries):
        if i and i % burst_size == 0:
            t += inter_gap
        elif i:
            t += intra_gap
        times.append(t)
    return times


def poisson_schedule(queries: int, mean_gap: float, seed: int) -> list[float]:
    """Arrival times with exponentially distributed gaps (mean mean_gap), seeded."""
    rng = random.Random(seed)
    t = 0.0
    times: list[float] = []
    for i in range(queries):
        if i:
            t += rng.expovariate(1.0 / mean_gap)
        times.append(t)
    return times


def warm_flags(arrivals: Sequence[float], ttl: float) -> list[bool]:
    """For each query after the first, whether the prefix was still warm on arrival."""
    ordered = sorted(arrivals)
    return [b - a <= ttl for a, b in zip(ordered, ordered[1:], strict=False)]


def warm_fraction(arrivals: Sequence[float], ttl: float) -> float | None:
    """f in the cost model: the share of queries after the first that find the prefix warm."""
    flags = warm_flags(arrivals, ttl)
    return (sum(flags) / len(flags)) if flags else None


def full_context_cost_schedule(
    price: ModelPrice,
    history_tokens: int,
    arrivals: Sequence[float],
    shape: QueryShape,
    ttl: float,
    cached: bool = True,
) -> float:
    """Arm A cost for one history under an explicit arrival schedule. With caching the first
    query writes the history and each later one reads it when it arrives within ttl of the
    previous query, otherwise it writes it again. Without caching every query pays the plain
    input rate."""
    if not arrivals:
        return 0.0
    prompt = history_tokens + shape.question_tokens
    r_in, r_cached, r_write, r_out = _rates(price, prompt)
    tail = shape.question_tokens * r_in + shape.answer_tokens * r_out
    if not cached:
        return len(arrivals) * (history_tokens * r_in + tail)
    warm = warm_flags(arrivals, ttl)
    cost = history_tokens * r_write + tail
    for is_warm in warm:
        cost += history_tokens * (r_cached if is_warm else r_write) + tail
    return cost


def crossover_for_spacing(
    price: ModelPrice,
    history_tokens: int,
    ingest_usd: float,
    context_tokens: int,
    shape: QueryShape,
    gap: float,
    ttl: float,
    cached: bool = True,
    retrieval_usd_per_query: float = 0.0,
    max_queries: int = 1000,
) -> int | None:
    """Smallest number of queries at which the memory arm is cheaper than full context when the
    queries arrive gap seconds apart, or None within max_queries (the tested range is 1 to 1,000).
    Spacing matters because a gap above the cache lifetime makes every full-context query
    rewrite the history."""
    for q in range(1, max_queries + 1):
        full = full_context_cost_schedule(price, history_tokens, fixed_schedule(q, gap), shape, ttl, cached)
        memory = memory_cost(price, ingest_usd, q, context_tokens, shape, retrieval_usd_per_query)
        if memory < full:
            return q
    return None


def crossover_by_spacing(
    price: ModelPrice,
    history_tokens: int,
    ingest_usd: float,
    context_tokens: int,
    shape: QueryShape,
    gaps: Sequence[float],
    ttl: float,
    retrieval_usd_per_query: float = 0.0,
    max_queries: int = 1000,
) -> dict[float, dict[str, int | None]]:
    """Crossover query count for each gap, with the breakpoint (cached) and without it."""
    return {
        gap: {
            mode: crossover_for_spacing(
                price, history_tokens, ingest_usd, context_tokens, shape, gap, ttl,
                cached=(mode == "cached"),
                retrieval_usd_per_query=retrieval_usd_per_query,
                max_queries=max_queries,
            )
            for mode in ("cached", "uncached")
        }
        for gap in gaps
    }


def measured_warm_fraction(records: Sequence[dict[str, Any]]) -> float | None:
    """f from logged cache fields: among arm A questions after the first on each history (in the
    order they were asked), the share whose reader call read cached tokens. Only meaningful for
    histories that carried a cache breakpoint."""
    by_history: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_history.setdefault(r["history_id"], []).append(r)
    hits = total = 0
    for rows in by_history.values():
        for r in sorted(rows, key=lambda r: r["ts"])[1:]:
            total += 1
            hits += r["reader"]["cached_tokens"] > 0
    return (hits / total) if total else None
