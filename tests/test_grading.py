"""Per-benchmark grading (prompt v6): LongMemEval's own judge prompts, Mem0's secondary LoCoMo grade,
MemoryAgentBench's per-source answer lines, native pass tests and summary grader."""

import pytest
from conftest import FakeClient, make_item, make_response

from memstudy import official_prompts as official
from memstudy.judge import Judge, parse_mem0_label, parse_yes_no, summary_scores
from memstudy.llm import ModelCaller
from memstudy.prompts import JUDGE_INSTRUCTIONS, judge_scheme, reader_task_key, task_key, uses_mem0_judge
from memstudy.reader import Reader
from memstudy.runner import secondary_scores, summarize_records
from memstudy.schema import Item
from memstudy.scoring import native_match, parse_label, parse_option


def _judge(prices, budget, cfg, texts):
    replies = iter(texts)
    client = FakeClient(lambda kw: make_response(text=next(replies), model="gpt-5-nano-test"))
    return client, Judge(ModelCaller(client, prices["gpt-5-nano"], budget), cfg)


def _lme(category, abstention=False, bench="longmemeval"):
    return Item(
        item_id="q1", bench=bench, history_id="q1", category=category, question="How many days?",
        gold="18", meta={"abstention": abstention},
    )


def _mab(source, gold="17", **meta):
    return Item(
        item_id="m1", bench="memoryagentbench", history_id="h", category="Long_Range_Understanding",
        question="q", gold=gold, meta={"source": source, "answers": [gold], **meta},
    )


def _text(call):
    return " ".join(part["text"] for msg in call["input"] for part in msg["content"])


def test_each_item_gets_the_grader_of_its_benchmark_never_of_its_arm():
    assert judge_scheme(_lme("temporal-reasoning")) == "longmemeval"
    assert judge_scheme(_lme("knowledge-update", bench="composite")) == "longmemeval"
    assert judge_scheme(_mab("infbench_sum_eng_shots2")) == "summary"
    assert judge_scheme(_mab("icl_banking77_5900shot_balance")) == "default"
    assert judge_scheme(make_item()) == "default"


@pytest.mark.parametrize(
    ("category", "abstention", "template"),
    [
        ("single-session-user", False, official.LME_DEFAULT),
        ("multi-session", False, official.LME_DEFAULT),
        ("temporal-reasoning", False, official.LME_TEMPORAL),
        ("knowledge-update", False, official.LME_KNOWLEDGE_UPDATE),
        ("single-session-preference", False, official.LME_PREFERENCE),
        ("multi-session", True, official.LME_ABSTENTION),
    ],
)
def test_longmemeval_items_are_graded_with_the_official_prompt_for_their_type(
    prices, budget, cfg, category, abstention, template
):
    client, judge = _judge(prices, budget, cfg, ["yes"])
    result = judge.grade(stage="pilot", tag={}, item=_lme(category, abstention), model_answer="19 days")
    sent = client.responses.calls[0]["input"]
    assert len(sent) == 1 and sent[0]["role"] == "user"  # no system text, as in the official script
    assert sent[0]["content"][0]["text"] == template.format("How many days?", "18", "19 days")
    assert result.correct is True and result.scheme == "longmemeval"


@pytest.mark.parametrize(
    ("text", "expected"), [("yes", True), ("No.", False), ("Yes, it is correct.", True), ("", None), ("maybe", None)]
)
def test_yes_no_parsing_follows_the_official_contains_yes_rule(text, expected):
    assert parse_yes_no(text) is expected


def test_locomo_answers_also_get_mem0s_grade_but_adversarial_questions_do_not(prices, budget, cfg):
    client, judge = _judge(prices, budget, cfg, ['Same topic. {"label": "CORRECT"}'])
    item = make_item()
    assert uses_mem0_judge(item)
    assert not uses_mem0_judge(item.model_copy(update={"meta": {"abstention": True}}))
    assert not uses_mem0_judge(_lme("multi-session"))
    result = judge.grade_mem0(stage="pilot", tag={}, item=item, model_answer="alpha")
    text = _text(client.responses.calls[0])
    assert f"Gold answer: {item.gold}" in text and "Generated answer: alpha" in text
    assert "generous with your grading" in text
    assert result.correct is True and result.scheme == "mem0"
    ruled = judge.grade_mem0(stage="pilot", tag={}, item=item, model_answer="Not mentioned")
    assert ruled.correct is False and len(client.responses.calls) == 1


@pytest.mark.parametrize(
    ("text", "expected"), [('{"label": "CORRECT"}', True), ('{"label":"WRONG"}', False), ("CORRECT", None)]
)
def test_mem0_label_parsing(text, expected):
    assert parse_mem0_label(text) is expected


def test_summary_scores_use_memoryagentbenchs_formula():
    scores = summary_scores(
        'ok {"fluency": 1}', 'so {"supported_key_points": [1, 2], "recall": 2}',
        '{"precision": 3, "sentence_count": 4}', keypoints=4,
    )
    assert scores == pytest.approx({"fluency": 1.0, "recall": 0.5, "precision": 0.75, "f1": 0.6})
    assert summary_scores('{"fluency": 0}', '{"recall": 4}', '{"precision": 4, "sentence_count": 4}', 4)["f1"] == 0
    assert summary_scores("no json", '{"recall": 1}', '{"precision": 1, "sentence_count": 1}', 4) is None


def test_a_summary_is_scored_with_three_calls_and_never_counted_as_unparseable(prices, budget, cfg):
    replies = ['{"fluency": 1}', '{"recall": 1}', '{"precision": 1, "sentence_count": 2}']
    client, judge = _judge(prices, budget, cfg, replies)
    item = _mab("infbench_sum_eng_shots2", gold="Expert summary.", keypoints=["a", "b"])
    result = judge.grade(stage="pilot", tag={}, item=item, model_answer="A summary.")
    assert len(client.responses.calls) == 3 and result.correct is None
    assert all(c["max_output_tokens"] == cfg["judge"]["summary_max_output_tokens"] for c in client.responses.calls)
    assert "1. a\n2. b" in _text(client.responses.calls[1]) and "Expert summary." in _text(client.responses.calls[2])
    assert result.scores == pytest.approx({"fluency": 1.0, "recall": 0.5, "precision": 0.5, "f1": 0.5})
    assert result.call.usage.input_tokens == 3 * 1000 and result.call.cost_usd > 0


def test_every_memoryagentbench_source_gets_its_answer_line_and_others_get_none(prices, budget, cfg):
    calls = []

    def handler(kw):
        calls.append(kw)
        return make_response(text="label: 17")

    reader = Reader(ModelCaller(FakeClient(handler), prices["gpt-6-luna"], budget), cfg)
    for key in ("icl_", "infbench_sum", None):
        reader.answer(
            stage="pilot", tag={}, context="c", question="text", question_date=None,
            cache_prefix=False, cache_key="k", task_key=key,
        )
    assert 'Only output "label: <number>"' in _text(calls[0]) and calls[0]["max_output_tokens"] == 512
    assert "1,000 to 1,200 words" in _text(calls[1]) and calls[1]["max_output_tokens"] == 2048
    assert "Task format" not in _text(calls[2])
    assert task_key(_mab("eventqa_65536")) == "eventqa" and task_key(make_item()) is None


def test_native_pass_tests_per_source():
    assert parse_label("label: 17") == "17" and parse_label("I think 5") == "5" and parse_label("none") is None
    assert native_match("icl_", "label: 17", ["17"])
    assert not native_match("icl_", "label: 17", ["1"])  # substring matching would pass this
    assert parse_option('{"answer": "C. The Brandt couple", "reasoning": "..."}') == "C"
    assert native_match("detective_qa", "C. The Brandts", ["C. The Brandt couple"])
    assert not native_match("detective_qa", "B. Mr. and Mrs. MacNamara", ["C. The Brandt couple"])
    assert native_match("eventqa", "Debbie tried on different dresses.", ["Debbie tried on different dresses"])
    assert native_match("infbench_sum", "summary", ["gold"]) is None


def test_secondary_scores_sit_beside_the_primary_accuracy():
    records = [
        {"correct_mem0": True, "metrics": {"native_match": True}},
        {"correct_mem0": False, "metrics": {"native_match": None}},
        {"summary_scores": {"f1": 0.4, "recall": 0.5, "precision": 0.3, "fluency": 1}, "metrics": None},
    ]
    s = secondary_scores(records)
    assert s["accuracy_mem0_judge"] == 0.5 and s["n_mem0_graded"] == 2
    assert s["native_match"] == 1.0 and s["n_native"] == 1
    assert s["summary_f1"] == 0.4 and s["n_summaries_scored"] == 1


def test_summary_items_without_a_verdict_are_not_judge_failures():
    def rec(scheme, correct, scores=None):
        usage = {
            "input_tokens": 1, "cached_tokens": 0, "cache_write_tokens": 0, "output_tokens": 1, "reasoning_tokens": 0
        }
        return {
            "reader": {"usage": usage, "cost_usd": 0.0, "model_returned": "m"},
            "retrieval": {"cost": {"usd": 0.0}}, "latency_s": {"query_total": 0.0, "retrieval": 0.0, "reader": 0.0},
            "judge": {"scheme": scheme, "usage": usage, "cost_usd": 0.0, "model_returned": "j"},
            "correct": correct, "summary_scores": scores, "category": "c", "metrics": None,
            "context_tokens_counted": 0, "retrieved": [],
        }
    out = summarize_records([rec("summary", None, {"f1": 1, "recall": 1, "precision": 1, "fluency": 1}),
                             rec("summary", None), rec("default", None)], [])
    assert out["n_judge_unparseable"] == 1 and out["n_summary_unparseable"] == 1


def test_a_question_already_wrapped_by_task_prompts_gets_no_second_task_line():
    wrapped = _mab("icl_banking77_5900shot_balance", task_prompt="icl_")
    assert reader_task_key(wrapped) is None and task_key(wrapped) == "icl_"  # still scored natively
    assert reader_task_key(_mab("icl_banking77_5900shot_balance")) == "icl_"


def test_judge_rule_needs_every_item_only_when_the_question_asks_for_all_of_them():
    assert "the question asks for all of them" in JUDGE_INSTRUCTIONS
    assert "an answer that gives at least one of them" in JUDGE_INSTRUCTIONS
