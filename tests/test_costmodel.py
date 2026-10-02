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


# --- query spacing -------------------------------------------------------------------------

from memstudy.costmodel import (  # noqa: E402
    burst_schedule,
    crossover_by_spacing,
    crossover_for_spacing,
    fixed_schedule,
    full_context_cost_schedule,
    measured_warm_fraction,
    poisson_schedule,
    ttl_seconds,
    warm_fraction,
)

TTL = ttl_seconds("30m")


def test_ttl_parsing():
    assert TTL == 1800 and ttl_seconds("1h") == 3600 and ttl_seconds("90s") == 90
    with pytest.raises(ValueError):
        ttl_seconds("soon")


def test_gaps_within_the_lifetime_match_the_closed_form(prices):
    luna = prices["gpt-6-luna"]
    sched = fixed_schedule(10, 60)
    assert full_context_cost_schedule(luna, 100_000, sched, SHAPE, TTL) == pytest.approx(
        full_context_cost(luna, 100_000, 10, SHAPE, cached=True)
    )


def test_gaps_beyond_the_lifetime_rewrite_the_history_every_time(prices):
    luna = prices["gpt-6-luna"]
    h, tail = 100_000, (50 * 0.10 + 10 * 0.50) / 1e6
    cost = full_context_cost_schedule(luna, h, fixed_schedule(5, 3600), SHAPE, TTL)
    assert cost == pytest.approx(5 * (h * 0.125 / 1e6 + tail))
    assert cost > full_context_cost_schedule(luna, h, fixed_schedule(5, 3600), SHAPE, TTL, cached=False)


def test_a_gap_exactly_at_the_lifetime_is_still_warm():
    assert warm_fraction([0, 1800, 3600], TTL) == 1.0
    assert warm_fraction([0, 1801], TTL) == 0.0
    assert warm_fraction([0], TTL) is None


def test_bursts_are_warm_inside_and_cold_between_when_idle_exceeds_the_lifetime():
    times = burst_schedule(9, burst_size=3, intra_gap=10, inter_gap=7200)
    assert times[:3] == [0, 10, 20] and times[3] == 7220
    assert warm_fraction(times, TTL) == pytest.approx(6 / 8)


def test_poisson_schedule_is_seeded_and_increasing():
    a, b = poisson_schedule(50, 600, seed=1), poisson_schedule(50, 600, seed=1)
    assert a == b and a == sorted(a) and a != poisson_schedule(50, 600, seed=2)


def test_sparse_traffic_moves_the_crossover_earlier_for_cached_full_context(prices):
    luna = prices["gpt-6-luna"]
    args = (luna, 100_000, 0.04, 7000, SHAPE)
    burst = crossover_for_spacing(*args, gap=60, ttl=TTL, cached=True)
    sparse = crossover_for_spacing(*args, gap=3600, ttl=TTL, cached=True)
    assert sparse is not None and (burst is None or sparse < burst)


def test_crossover_table_reports_both_cache_policies(prices):
    table = crossover_by_spacing(prices["gpt-6-luna"], 100_000, 0.04, 7000, SHAPE, [60, 3600], TTL)
    assert set(table) == {60, 3600} and set(table[60]) == {"cached", "uncached"}
    assert table[60]["uncached"] == table[3600]["uncached"]  # no cache, so spacing cannot matter


def test_warm_fraction_from_logged_cache_fields():
    def rec(h, ts, cached):
        return {"history_id": h, "ts": ts, "reader": {"cached_tokens": cached}}

    records = [rec("a", 1, 0), rec("a", 2, 900), rec("a", 3, 900), rec("b", 1, 0), rec("b", 2, 0)]
    assert measured_warm_fraction(records) == pytest.approx(2 / 3)
    assert measured_warm_fraction([rec("a", 1, 0)]) is None
