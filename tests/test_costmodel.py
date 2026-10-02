import pytest

from memstudy.costmodel import QueryShape, crossover_queries, full_context_cost, memory_cost

SHAPE = QueryShape(question_tokens=50, answer_tokens=10)


def test_cached_full_context_pays_one_write_then_reads(prices):
    luna = prices["gpt-6-luna"]
    h = 100_000
    one = full_context_cost(luna, h, 1, SHAPE, cached=True)
    ten = full_context_cost(luna, h, 10, SHAPE, cached=True)
    tail = (50 * 0.10 + 10 * 0.50) / 1e6
    assert one == pytest.approx(h * 0.125 / 1e6 + tail)
    assert ten == pytest.approx(h * 0.125 / 1e6 + 9 * (h * 0.01 / 1e6) + 10 * tail)


def test_uncached_full_context_scales_with_queries(prices):
    luna = prices["gpt-6-luna"]
    assert full_context_cost(luna, 1000, 4, SHAPE, False) == pytest.approx(
        4 * full_context_cost(luna, 1000, 1, SHAPE, False)
    )


def test_long_context_history_costs_more_per_token(prices):
    luna = prices["gpt-6-luna"]
    below = full_context_cost(luna, 271_000, 1, SHAPE, False)
    above = full_context_cost(luna, 273_000, 1, SHAPE, False)
    assert above > 2 * below * 0.99


def test_crossover_exists_when_memory_is_cheaper_per_query(prices):
    luna = prices["gpt-6-luna"]
    q = crossover_queries(luna, 200_000, ingest_usd=0.05, context_tokens=7000, shape=SHAPE, cached=False)
    assert q is not None and q > 0
    a = full_context_cost(luna, 200_000, int(q) + 2, SHAPE, False)
    b = memory_cost(luna, 0.05, int(q) + 2, 7000, SHAPE)
    assert b < a


def test_no_crossover_when_cached_reads_beat_retrieved_context(prices):
    luna = prices["gpt-6-luna"]
    # cached reads of 1000 tokens cost less per query than 7000 uncached retrieved tokens
    assert crossover_queries(luna, 1000, 0.01, 7000, SHAPE, cached=True) is None
