"""Confidence intervals, paired tests, LoCoMo diagnostics, latency split and run provenance."""

import json
from pathlib import Path

import pytest

from evals.analysis.aggregate import (
    MIN_PILOT_ACCOUNTS,
    aggregate_records,
    confidence_interval_report,
    locomo_diagnostics,
    mcnemar_report,
)
from evals.runners.checkpoint import RecordCheckpoint, write_run_json
from evals.runners.providers import TimedEmbedder
from evals.runners.run_external import run_external
from evals.schemas import EvaluationRecord


def rec(**updates):
    fields = dict(
        run_id="r", dataset="cross_scope_mem", system="scopegraph", scenario_id="a0",
        question_id="q0", question_type="recall", question="Q?", gold_answer="X",
        config_hash="h", seed=42,
    )
    fields.update(updates)
    return EvaluationRecord(**fields)


def paired(accounts: int, per_account: int = 4, full_wins: bool = True):
    """Full avoids contamination (0) while the control always contaminates (1)."""
    rows = []
    for account in range(accounts):
        for question in range(per_account):
            common = dict(
                scenario_id=f"a{account}", question_id=f"q{question}",
                gold_scope_ids=["p"], allowed_scope_ids=["global", "p"],
                gold_source_ids=["m"], retrieved_source_ids=[["m"]],
                answer_evaluated=True, answer="X",
            )
            rows.append(rec(ablation="full", retrieved_origin_scope_ids=[["p"]], **common))
            rows.append(rec(
                ablation="vector_only_control",
                retrieved_origin_scope_ids=[["other"] if full_wins else ["p"]], **common,
            ))
    return rows


def test_single_account_has_no_degenerate_interval():
    summary = aggregate_records(paired(1)[:: 2])
    assert summary["scopegraph"]["account_count"] == 1
    assert "cross_scope_contamination_ci95_low" not in summary["scopegraph"]
    report = confidence_interval_report(paired(1))
    system = report["systems"]["scopegraph"]
    assert system["underpowered"] is True
    metric = system["metrics"]["cross_scope_contamination"]
    assert metric["ci95_low"] is None and metric["ci95_high"] is None


def test_intervals_resample_accounts_and_flag_small_pilots():
    few = confidence_interval_report(paired(MIN_PILOT_ACCOUNTS - 1))
    enough = confidence_interval_report(paired(MIN_PILOT_ACCOUNTS))
    assert few["systems"]["scopegraph"]["underpowered"] is True
    assert enough["systems"]["scopegraph"]["underpowered"] is False
    assert enough["systems"]["scopegraph"]["n_accounts"] == MIN_PILOT_ACCOUNTS
    assert enough["protocol"]["resampling_unit"].startswith("account")


def test_difference_sign_is_full_minus_control_and_favours_lower_contamination():
    report = confidence_interval_report(paired(MIN_PILOT_ACCOUNTS))
    comparison = report["paired_differences_full_minus_control"][
        "cross_scope_mem/scopegraph/vector_only_control"
    ]
    metric = comparison["metrics"]["cross_scope_contamination"]
    assert metric["mean_difference"] == -1.0  # full (0) minus control (1)
    assert metric["lower_is_better"] is True
    assert metric["favours"] == "full"
    assert metric["accounts_negative"] == MIN_PILOT_ACCOUNTS
    assert "full minus control" in report["protocol"]["sign_convention"]


def test_mcnemar_only_covers_binary_outcomes():
    rows = mcnemar_report(paired(MIN_PILOT_ACCOUNTS))
    metrics = {row["metric"] for row in rows}
    assert "any_cross_scope_contamination" in metrics
    assert "recall_at_8" not in metrics and "stale_memory_error_rate" not in metrics
    row = next(r for r in rows if r["metric"] == "any_cross_scope_contamination")
    assert row["full_correct_control_wrong"] == 0
    assert row["control_correct_full_wrong"] == MIN_PILOT_ACCOUNTS * 4
    assert row["exact_p"] < 1e-6


def test_latency_stages_are_reported_separately():
    summary = aggregate_records([rec(
        retrieval_latency_ms=100.0, retrieval_embedding_ms=70.0, answer_latency_ms=1000.0,
        judge_latency_ms=500.0, embedding_preparation_ms=5.0,
    )])["scopegraph"]
    assert summary["retrieval_p50_ms"] == 100.0
    assert summary["retrieval_embedding_p50_ms"] == 70.0
    assert summary["retrieval_core_p50_ms"] == 30.0
    assert summary["answer_p50_ms"] == 1000.0 and summary["judge_p50_ms"] == 500.0


def locomo(question_id, category, score, gold, retrieved, delivered, conversation="c1"):
    return rec(
        dataset="locomo", scenario_id=conversation, question_id=question_id,
        benchmark_metadata={"category": category}, official_metric="llm_judge_accuracy",
        official_score=score, official_secondary_scores={"locomo_f1": score},
        gold_source_ids=gold, retrieved_source_ids=[[s] for s in retrieved],
        delivered_source_ids=[[s] for s in delivered],
    )


def test_locomo_diagnostics_split_category_5_and_gold_recall():
    rows = [
        locomo("1", "1", 1.0, ["a", "b"], ["a", "b"], ["a", "b"]),
        locomo("2", "1", 0.0, ["a", "b"], ["a"], ["a"]),
        locomo("3", "2", 0.0, ["c"], [], []),
        locomo("4", "5", 1.0, ["d"], [], []),
    ]
    report = locomo_diagnostics(rows)["scopegraph/full"]
    assert report["all_categories"]["judge_accuracy"]["mean"] == pytest.approx(0.5)
    assert report["excluding_category_5"]["judge_accuracy"]["mean"] == pytest.approx(1 / 3)
    assert report["excluding_category_5"]["judge_accuracy"]["n_questions"] == 3
    one = report["categories"]["1"]
    assert one["gold_source_recall_retrieved"] == pytest.approx(0.75)
    assert one["all_gold_delivered_rate"] == pytest.approx(0.5)
    assert one["accuracy_when_all_gold_delivered"] == 1.0
    assert one["accuracy_when_gold_partly_or_not_delivered"] == 0.0
    assert report["categories"]["2"]["outcome_stages"] == {"wrong_gold_not_retrieved": 1}
    assert report["categories"]["5"]["scored_by"] == "abstention rule"
    assert report["all_categories"]["judge_accuracy"]["underpowered"] == 1.0


def test_bulk_text_goes_to_sidecar_only_for_new_runs(tmp_path: Path):
    bulky = rec(retrieved_source_contents=[["t"]], delivered_source_contents=[["t"]])
    new = RecordCheckpoint(tmp_path / "new.jsonl", resume=False)
    new.append(bulky)
    main = json.loads((tmp_path / "new.jsonl").read_text())
    assert main["retrieved_source_contents"] == [] and main["delivered_source_contents"] == []
    side = json.loads((tmp_path / "new.bulk.jsonl").read_text())
    assert side["retrieved_source_contents"] == [["t"]] and side["question_id"] == "q0"

    legacy_path = tmp_path / "old.jsonl"
    legacy_path.write_text(rec(question_id="old").model_dump_json() + "\n")
    legacy = RecordCheckpoint(legacy_path, resume=True)
    legacy.append(bulky)
    assert not (tmp_path / "old.bulk.jsonl").exists()
    assert json.loads(legacy_path.read_text().splitlines()[1])["retrieved_source_contents"] == [
        ["t"]
    ]


def test_run_json_records_provenance_and_merges_updates(tmp_path: Path):
    path = write_run_json(tmp_path / "run.jsonl", run_id="r1", status="running", seed=42)
    assert path.name == "run.run.json"
    write_run_json(tmp_path / "run.jsonl", status="complete")
    data = json.loads(path.read_text())
    assert data["status"] == "complete" and data["run_id"] == "r1"
    assert {"commit", "dirty", "working_diff_sha256"} <= set(data["git"])
    batch = write_run_json(tmp_path / "batch", kind="batch")
    assert batch == tmp_path / "batch" / "run.json"


@pytest.mark.asyncio
async def test_timed_embedder_accumulates_and_resets():
    class Fake:
        model_name = "fake"

        async def embed(self, texts):
            return [[1.0] for _ in texts]

    timed = TimedEmbedder(Fake())
    await timed.embed(["a"])
    assert timed.elapsed_ms > 0
    timed.reset()
    assert timed.elapsed_ms == 0.0


LOCOMO_SAMPLE = [{
    "sample_id": "c1",
    "conversation": {
        "session_1_date_time": "2024-01-01",
        "session_1": [{"speaker": "A", "dia_id": "D1", "text": "We use Neo4j."}],
    },
    "qa": [{"question": "Which database?", "answer": "Neo4j", "category": "1",
            "evidence": ["D1"]}],
}]


@pytest.mark.asyncio
async def test_locomo_vector_only_amendment_is_labelled_and_restricted(tmp_path: Path):
    data = tmp_path / "locomo.json"
    data.write_text(json.dumps(LOCOMO_SAMPLE))
    out = await run_external(
        dataset="locomo", path=data, system_name="scopegraph", output=tmp_path / "v.jsonl",
        storage="memory", ablation="vector_only",
    )
    record = json.loads(out.read_text())
    assert record["ablation"] == "vector_only"
    assert record["retrieval_embedding_ms"] is not None
    run = json.loads(out.with_suffix(".run.json").read_text())
    assert run["ablation"] == "vector_only" and "amendment" in run["protocol"]
    assert run["status"] == "complete"
    full = await run_external(
        dataset="locomo", path=data, system_name="scopegraph", output=tmp_path / "f.jsonl",
        storage="memory",
    )
    assert json.loads(full.read_text())["config_hash"] != record["config_hash"]
    with pytest.raises(ValueError, match="LoCoMo-only"):
        await run_external(
            dataset="longmemeval", path=data, system_name="scopegraph",
            output=tmp_path / "x.jsonl", storage="memory", ablation="vector_only",
        )


def test_locomo_terminal_summary_logs_category_gold_recall_and_without_category_5():
    from evals.runners.run_external import _terminal_summary

    text = _terminal_summary([
        locomo("1", "1", 1.0, ["a"], ["a"], ["a"]),
        locomo("2", "5", 0.0, ["b"], [], []),
    ])
    assert "(all categories)" in text and "(without category 5)" in text
    assert "category 1 (n=1): accuracy 100.0% | gold-source recall retrieved 100.0%" in text
    assert "category 5 (n=1)" in text and "no gold retrieved 100.0%" in text


@pytest.mark.asyncio
async def test_new_runs_use_timestamp_names_and_write_a_readable_report(tmp_path: Path):
    import re

    from evals.runners.run_all import run_all

    paths = await run_all(
        scenario_count=1, difficulty=1, output=str(tmp_path), profile="smoke", storage="memory",
    )
    batch = Path(paths[0]).parent
    assert re.fullmatch(r"\d\d_\d\d__\d\d_\d\d", batch.name)
    text = (batch / "report.md").read_text()
    assert "Full ScopeGraph compared with each control" in text
    assert "Pulled in other scopes' memories" in text
    header = (batch / "questions.csv").read_text().splitlines()[0]
    assert header.startswith("condition,account,question_id")

    data = tmp_path / "locomo.json"
    data.write_text(json.dumps(LOCOMO_SAMPLE))
    out = await run_external(
        dataset="locomo", path=data, system_name="scopegraph", output=tmp_path / "ext",
        storage="memory",
    )
    assert re.fullmatch(r"\d\d_\d\d__\d\d_\d\d\.jsonl", out.name)
    assert (tmp_path / "ext" / f"{out.stem}.report.md").exists()
    assert (tmp_path / "ext" / f"{out.stem}.questions.csv").exists()
