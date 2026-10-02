import dataclasses

import pytest
from conftest import FakeClient, make_response

from memstudy.budget import (
    Budget,
    BudgetExceeded,
    UnverifiedPrice,
    cost_usd,
    worst_case_usd,
)
from memstudy.llm import ModelCaller
from memstudy.schema import Usage


def test_cost_splits_uncached_cached_and_write(prices):
    luna = prices["gpt-6-luna"]
    usage = Usage(input_tokens=1000, cached_tokens=400, cache_write_tokens=100, output_tokens=50)
    expected = (500 * 0.10 + 400 * 0.01 + 100 * 0.125 + 50 * 0.50) / 1e6
    assert cost_usd(luna, usage) == pytest.approx(expected)


def test_reasoning_tokens_are_not_added_twice(prices):
    nano = prices["gpt-5-nano"]
    with_reasoning = Usage(input_tokens=100, output_tokens=100, reasoning_tokens=60)
    without = Usage(input_tokens=100, output_tokens=100)
    assert cost_usd(nano, with_reasoning) == cost_usd(nano, without)
    assert cost_usd(nano, without) == pytest.approx((100 * 0.05 + 100 * 0.40) / 1e6)


def test_long_context_multiplier_applies_to_whole_request(prices):
    luna = prices["gpt-6-luna"]
    short = Usage(input_tokens=272_000, output_tokens=100)
    long = Usage(input_tokens=272_001, output_tokens=100)
    assert cost_usd(luna, short) == pytest.approx((272_000 * 0.10 + 100 * 0.50) / 1e6)
    assert cost_usd(luna, long) == pytest.approx((272_001 * 0.20 + 100 * 0.75) / 1e6)


def test_unverified_price_is_refused(prices):
    mini = dataclasses.replace(prices["gpt-6-luna"], verified=False)
    with pytest.raises(UnverifiedPrice):
        cost_usd(mini, Usage(input_tokens=10))
    with pytest.raises(UnverifiedPrice):
        worst_case_usd(mini, 10, 10)


def test_worst_case_covers_cache_write_rate(prices):
    luna = prices["gpt-6-luna"]
    assert worst_case_usd(luna, 1000, 100) == pytest.approx((1000 * 0.125 + 100 * 0.50) / 1e6)


def test_guard_refuses_over_stage_cap(tmp_path):
    b = Budget(tmp_path / "l.jsonl", total_cap=350, stage_caps={"pilot": 15.0})
    b.charge("pilot", 14.5)
    b.guard("pilot", 0.5)
    with pytest.raises(BudgetExceeded):
        b.guard("pilot", 0.51)


def test_guard_refuses_over_total_cap(tmp_path):
    b = Budget(tmp_path / "l.jsonl", total_cap=20, stage_caps={"pilot": 15.0, "chat": 90.0})
    b.charge("pilot", 12)
    b.charge("chat", 7)
    with pytest.raises(BudgetExceeded):
        b.guard("chat", 1.5)


def test_default_caps_match_the_preregistered_budget(tmp_path):
    b = Budget(tmp_path / "l.jsonl")
    assert b.total_cap == 350 and b.stage_caps == {"pilot": 15, "chat": 90}


def test_ledger_resume_restores_spend(tmp_path):
    path = tmp_path / "l.jsonl"
    first = Budget(path)
    first.charge("pilot", 3.0, note="x")
    first.charge("chat", 4.0)
    resumed = Budget(path)
    assert resumed.spent("pilot") == 3.0 and resumed.spent_total == 7.0
    assert resumed.remaining("pilot") == 12.0


def test_unknown_stage_is_refused(tmp_path):
    with pytest.raises(BudgetExceeded):
        Budget(tmp_path / "l.jsonl").guard("mystery", 0.0)


def test_guard_blocks_the_call_before_it_is_made(prices, tmp_path):
    client = FakeClient(lambda kw: make_response())
    tiny = Budget(tmp_path / "l.jsonl", total_cap=350, stage_caps={"pilot": 0.000001})
    caller = ModelCaller(client, prices["gpt-6-luna"], tiny)
    big = [{"role": "user", "content": [{"type": "input_text", "text": "word " * 5000}]}]
    with pytest.raises(BudgetExceeded):
        caller.call(
            stage="pilot",
            tag={},
            input_items=big,
            max_output_tokens=512,
            reasoning_effort="none",
        )
    assert client.responses.calls == []


def test_worst_case_never_below_actual_cost(prices):
    luna = prices["gpt-6-luna"]
    usage = Usage(input_tokens=5000, cache_write_tokens=5000, output_tokens=200)
    assert worst_case_usd(luna, 5000, 200) >= cost_usd(luna, usage)
