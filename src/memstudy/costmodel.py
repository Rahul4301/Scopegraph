"""Closed-form cost per history for each arm, with queries per history (Q) as a variable.

All inputs are measured token counts and the prices in configs/prices.yaml. Used for the pilot
projection (Phase 2) and the cost crossover analysis (Phase 4).
"""

from __future__ import annotations

from dataclasses import dataclass

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
