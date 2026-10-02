import json

import pytest
from conftest import FakeClient, make_item, make_response

from memstudy.budget import cost_usd
from memstudy.judge import Judge, parse_verdict
from memstudy.llm import IncompleteResponse, ModelCaller, usage_from_response
from memstudy.reader import Reader, build_input


def _caller(prices, budget, handler, model="gpt-6-luna"):
    client = FakeClient(handler)
    return client, ModelCaller(client, prices[model], budget)


def test_usage_reads_cached_and_cache_write_and_reasoning():
    usage, missing = usage_from_response(
        make_response(input_tokens=15000, cached=12000, written=3000, output=40, reasoning=25).usage
    )
    assert (usage.cached_tokens, usage.cache_write_tokens) == (12000, 3000)
    assert (usage.output_tokens, usage.reasoning_tokens) == (40, 25)
    assert missing == []


def test_missing_cache_fields_are_reported_not_hidden():
    raw = make_response().usage
    raw.input_tokens_details = None
    raw.output_tokens_details = None
    usage, missing = usage_from_response(raw)
    assert usage.cached_tokens == 0
    assert set(missing) == {"cached_tokens", "cache_write_tokens", "reasoning_tokens"}


def test_reader_logs_cached_tokens_and_charges_each_class(prices, budget, cfg):
    handler = lambda kw: make_response(  # noqa: E731
        input_tokens=20000, cached=12800, written=7000, output=30, reasoning=0
    )
    client, caller = _caller(prices, budget, handler)
    result = Reader(caller, cfg).answer(
        stage="pilot",
        tag={"arm": "A"},
        context="history text",
        question="q?",
        question_date=None,
        cache_prefix=True,
        cache_key="A-locomo_h1",
    )
    assert result.usage.cached_tokens == 12800
    expected = (200 * 0.10 + 12800 * 0.01 + 7000 * 0.125 + 30 * 0.50) / 1e6
    assert result.cost_usd == pytest.approx(expected)
    row = json.loads(budget.ledger_path.read_text().splitlines()[-1])
    assert row["cached_tokens"] == 12800 and row["cache_write_tokens"] == 7000
    assert row["arm"] == "A" and row["purpose"] == "read"
    assert budget.spent("pilot") == pytest.approx(expected)


def test_prefix_arm_sets_one_breakpoint_after_context_and_question_last():
    items = build_input("CTX", "Q?", "2023/05/30", True)
    user = items[1]["content"]
    assert user[0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert "CTX" in user[0]["text"] and "Q?" in user[1]["text"]
    assert "prompt_cache_breakpoint" not in user[1]
    assert "Current date: 2023/05/30" in user[1]["text"]


def test_memory_arms_send_no_breakpoint_so_they_pay_no_write_surcharge():
    items = build_input("CTX", "Q?", None, False)
    assert all("prompt_cache_breakpoint" not in b for m in items for b in m["content"])


def test_reader_request_is_pinned_and_explicit(prices, budget, cfg):
    client, caller = _caller(prices, budget, lambda kw: make_response())
    Reader(caller, cfg).answer(
        stage="pilot",
        tag={},
        context="c",
        question="q",
        question_date=None,
        cache_prefix=True,
        cache_key="k",
    )
    sent = client.responses.calls[0]
    assert sent["model"] == "gpt-6-luna"
    assert sent["temperature"] == 0
    assert sent["reasoning"] == {"effort": "none"}
    assert sent["store"] is False
    assert sent["extra_body"]["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert sent["extra_body"]["prompt_cache_key"] == "k"


def test_incomplete_response_raises_but_cost_is_still_charged(prices, budget, cfg):
    client, caller = _caller(prices, budget, lambda kw: make_response(status="incomplete"))
    with pytest.raises(IncompleteResponse):
        Reader(caller, cfg).answer(
            stage="pilot", tag={}, context="c", question="q", question_date=None,
            cache_prefix=False, cache_key="k",
        )
    assert budget.spent("pilot") > 0


def test_judge_logs_reasoning_tokens_and_uses_the_configured_effort(prices, budget, cfg):
    handler = lambda kw: make_response(  # noqa: E731
        text='{"verdict": "CORRECT"}', input_tokens=300, output=64, reasoning=48
    )
    client, caller = _caller(prices, budget, handler, "gpt-5-nano")
    result = Judge(caller, cfg).grade(
        stage="pilot", tag={"arm": "A"}, item=make_item(), model_answer="alpha"
    )
    sent = client.responses.calls[0]
    assert sent["reasoning"] == {"effort": cfg["judge"]["reasoning_effort"]}
    assert "temperature" not in sent
    assert sent["max_output_tokens"] == cfg["judge"]["max_output_tokens"]
    assert result.correct is True and result.call.usage.reasoning_tokens == 48
    expected = (300 * 0.05 + 64 * 0.40) / 1e6
    assert cost_usd(prices["gpt-5-nano"], result.call.usage) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"verdict": "CORRECT"}', True),
        ('{"verdict":"incorrect"}', False),
        ('Sure! {"verdict": "INCORRECT"}', False),
        ("CORRECT", None),
        ("", None),
    ],
)
def test_verdict_parsing(text, expected):
    assert parse_verdict(text) is expected


def test_not_mentioned_is_graded_incorrect_without_a_call_unless_the_item_is_an_abstention(
    prices, budget, cfg
):
    client, caller = _caller(prices, budget, lambda kw: make_response(text='{"verdict": "CORRECT"}'), "gpt-5-nano")
    judge = Judge(caller, cfg)
    ruled = judge.grade(stage="pilot", tag={}, item=make_item(), model_answer="Not mentioned.")
    assert ruled.correct is False and not client.responses.calls
    assert ruled.call.cost_usd == 0.0 and ruled.call.model_returned == "rule:not_mentioned"
    abstain = make_item().model_copy(update={"meta": {"abstention": True}})
    asked = judge.grade(stage="pilot", tag={}, item=abstain, model_answer="Not mentioned")
    assert asked.correct is True and len(client.responses.calls) == 1
