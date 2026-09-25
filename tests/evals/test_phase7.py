import json

import pytest

from evals.analysis.aggregate import (
    aggregate_records,
    paired_ablation_comparisons,
    score_record,
)
from evals.metrics.answer_accuracy import exact_match, token_f1
from evals.metrics.retrieval_precision import precision_at_k
from evals.metrics.retrieval_recall import recall_at_k
from evals.scenarios.generate_cross_scope import generate_cross_scope_mem
from evals.schemas import EvaluationRecord


def test_cross_scope_generator_is_deterministic() -> None:
    left = generate_cross_scope_mem(seed=42, difficulty=2)
    right = generate_cross_scope_mem(seed=42, difficulty=2)
    assert left.model_dump() == right.model_dump()
    assert {scope.id for scope in left.scopes} == {
        "global", "scope_alpha", "scope_beta", "scope_gamma"
    }
    assert any(example.question_type == "session_override" for example in left.examples)


def test_retrieval_and_answer_metrics() -> None:
    assert precision_at_k(["a", "b", "c"], ["b"], 2) == 0.5
    assert recall_at_k(["a", "b", "c"], ["b"], 2) == 1.0
    assert exact_match("MongoDB", "MongoDB") == 1.0
    assert token_f1("Beta uses MongoDB", "MongoDB") > 0.0


@pytest.mark.asyncio
async def test_raw_record_can_be_scored_and_aggregated() -> None:
    record = EvaluationRecord(
        run_id="run", dataset="cross_scope_mem", system="scopegraph", scenario_id="s",
        question_id="q", question_type="scope_specific_recall", question="What DB?",
        gold_answer="MongoDB", gold_scope_ids=["scope_beta"], gold_memory_ids=["m1"],
        retrieved_memory_ids=["m1"], retrieved_scope_ids=["scope_beta"],
        retrieved_statuses=["active"], answer="MongoDB", config_hash="hash", seed=42,
    )
    scored = score_record(record)
    assert scored.metrics["recall_at_8"] == 1.0
    assert aggregate_records([record])["scopegraph"]["query_count"] == 1.0


def test_record_is_jsonl_serializable() -> None:
    record = EvaluationRecord(
        run_id="run", dataset="cross_scope_mem", system="scopegraph", scenario_id="s",
        question_id="q", question_type="global_recall", question="What?", gold_answer="x",
        config_hash="hash", seed=42,
    )
    assert json.loads(record.model_dump_json())["system"] == "scopegraph"


def test_paired_ablation_comparison_uses_matching_questions() -> None:
    full = EvaluationRecord(
        run_id="full", dataset="cross_scope_mem", system="scopegraph", ablation="full",
        scenario_id="account", question_id="q", question_type="recall", question="What?",
        gold_answer="x", answer="x", answer_evaluated=True, config_hash="full", seed=42,
    )
    ablated = full.model_copy(
        update={
            "run_id": "ablated",
            "ablation": "no_scope_hierarchy",
            "answer": "wrong",
            "config_hash": "ablated",
        }
    )
    comparison = paired_ablation_comparisons([full, ablated])
    values = comparison["cross_scope_mem/scopegraph/full-minus-no_scope_hierarchy"]
    assert values["paired_question_count"] == 1
    assert values["exact_match_mean_difference"] == 1
    assert values["exact_match_mcnemar_full_wins"] == 1
