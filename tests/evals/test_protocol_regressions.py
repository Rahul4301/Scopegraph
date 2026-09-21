import asyncio
import json

import pytest

from evals.adapters.cross_scope_mem import KeywordEmbeddingProvider, ScenarioExtractor
from evals.analysis.aggregate import aggregate_records, confidence_intervals, score_record
from evals.analysis.scope_classification import evaluate_scope_classification
from evals.runners.providers import EvaluationProviders, PreparedEmbedder
from evals.runners.run_all import run_all
from evals.runners.run_eval import run_scenario
from evals.scenarios.generate_cross_scope import candidates_by_message, generate_cross_scope_mem
from evals.scenarios.research import generate_research_scenario
from evals.schemas import EvaluationRecord
from scopegraph.memory.retriever import RetrievalConfig


def record(**updates):
    fields = dict(
        run_id="run",
        dataset="test",
        system="scopegraph",
        scenario_id="s",
        question_id="q",
        question_type="recall",
        question="Database?",
        gold_answer="PostgreSQL",
        config_hash="hash",
        seed=42,
    )
    fields.update(updates)
    return EvaluationRecord(**fields)


@pytest.mark.parametrize("difficulty", [1, 2, 3, 4])
def test_gold_evidence_exists_at_question_time(difficulty):
    scenario = generate_cross_scope_mem(difficulty=difficulty)
    messages = {
        message.id: message for session in scenario.sessions for message in session.messages
    }
    for example in scenario.examples:
        assert example.gold_source_ids
        assert all(
            messages[source].timestamp <= example.timestamp for source in example.gold_source_ids
        )
    named = next(item for item in scenario.examples if item.question_id == "q_named_beta")
    assert named.gold_answer == ("PostgreSQL" if difficulty >= 3 else "MongoDB")


def test_missing_gold_is_not_perfect_recall():
    assert score_record(record()).metrics["recall_at_8"] == 0
    assert score_record(record(gold_source_ids=["missing"])).metrics["recall_at_8"] == 0


def test_provenance_scoring_survives_paraphrases_and_split_memories():
    result = score_record(
        record(
            gold_source_ids=["message"],
            retrieved_source_ids=[["message"], ["message"]],
            retrieved_memory_ids=["random1", "random2"],
            retrieved_contents=["The database is PostgreSQL", "Postgres is used"],
        )
    )
    assert result.metrics["recall_at_8"] == 1
    assert result.metrics["precision_at_8"] == 1


def test_scope_leakage_uses_origin_not_flattened_storage():
    result = score_record(
        record(
            gold_scope_ids=["beta"],
            allowed_scope_ids=["global", "beta"],
            retrieved_scope_ids=["global", "global"],
            retrieved_origin_scope_ids=[["global"], ["alpha"]],
        )
    )
    assert result.metrics["cross_scope_contamination"] == 0.5
    assert "scope_classification_accuracy" not in result.metrics


def test_offline_runs_do_not_claim_answer_accuracy_or_api_token_counts():
    result = aggregate_records([record(answer_evaluated=False)])
    assert "exact_match" not in result["scopegraph"]
    assert "answer_tokens_mean" not in result["scopegraph"]


def test_reports_reject_mixed_or_duplicate_runs():
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_records([record(), record()])
    with pytest.raises(ValueError, match="incompatible"):
        aggregate_records([record(), record(system="flat_graph", evaluation_mode="live")])
    with pytest.raises(ValueError, match="identical question"):
        confidence_intervals([record(), record(system="flat_graph", question_id="other")])


@pytest.mark.asyncio
async def test_replay_excludes_future_update_and_preserves_override():
    scenario = generate_cross_scope_mem(difficulty=3)
    providers = EvaluationProviders(
        ScenarioExtractor(candidates_by_message(scenario)),
        PreparedEmbedder(KeywordEmbeddingProvider()),
    )
    rows = await run_scenario(
        scenario,
        system_name="scopegraph",
        run_id="run",
        config={},
        config_hash="test",
        providers=providers,
    )
    by_id = {row.question_id: row for row in rows}
    override = by_id["q_beta_override"]
    assert "msg_session_beta_update" not in {
        source for sources in override.retrieved_source_ids for source in sources
    }
    assert score_record(override).metrics["recall_at_8"] == 1
    normal = by_id["q_beta_normal"]
    assert "msg_session_beta_override" not in {
        source for sources in normal.retrieved_source_ids for source in sources
    }
    assert score_record(normal).metrics["recall_at_8"] == 1
    assert all(not row.answer_evaluated and row.input_tokens is None for row in rows)


@pytest.mark.asyncio
async def test_batch_artifact_is_shared_and_reportable(tmp_path):
    paths = await run_all(
        systems=["vector_memory", "scopegraph"],
        scenario_count=1,
        difficulty=3,
        output=str(tmp_path),
        profile="smoke",
    )
    from pathlib import Path

    batch = Path(paths[0]).parent
    manifest = json.loads((batch / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["paths"] == paths
    assert (batch / "extractions.json").exists()
    classification = json.loads((batch / "classification.json").read_text())
    assert classification["evaluated"] is False
    assert classification["summary"]["scope_level_accuracy"] is None
    rows = []
    for path in paths:
        text = await asyncio.to_thread(Path(path).read_text)
        rows.extend(EvaluationRecord.model_validate_json(line) for line in text.splitlines())
    assert len({row.config_hash for row in rows}) == 1
    assert aggregate_records(rows)["scopegraph"]["recall_at_8"] == 1


def test_config_preserves_historical_terms_by_default():
    assert "previously" in RetrievalConfig.from_config({}).historical_query_terms
    assert RetrievalConfig.from_config({"historical_query_terms": []}).historical_query_terms == ()


@pytest.mark.asyncio
async def test_research_historical_question_uses_an_as_of_snapshot():
    scenario = generate_research_scenario(seed=42, difficulty=3)
    providers = EvaluationProviders(
        ScenarioExtractor(candidates_by_message(scenario)),
        PreparedEmbedder(KeywordEmbeddingProvider()),
    )
    rows = await run_scenario(
        scenario,
        system_name="scopegraph",
        run_id="run",
        config={},
        config_hash="test",
        providers=providers,
    )
    historical = next(row for row in rows if row.question_id == "q_beta_history")
    assert score_record(historical).metrics["recall_at_8"] == 1


def test_live_scope_classification_records_confusion_missing_and_extra_candidates():
    scenario = generate_research_scenario(seed=42, difficulty=3)
    predicted = candidates_by_message(scenario)
    override = predicted["msg_beta_override"][0]
    predicted["msg_beta_override"] = [
        override.model_copy(update={"proposed_scope_level": "scope", "durability": 1.0})
    ]
    predicted.pop("msg_global_language")
    predicted["msg_global_db"].append(
        predicted["msg_global_db"][0].model_copy(deep=True)
    )
    evaluation = evaluate_scope_classification(
        [scenario],
        {scenario.scenario_id: predicted},
        live_extraction=True,
    )
    assert evaluation.evaluated is True
    assert evaluation.summary["missing_candidate_count"] == 1
    assert evaluation.summary["extra_candidate_count"] == 1
    assert evaluation.summary["scope_level_accuracy"] < 1
    assert evaluation.confusion_matrix["session"]["scope"] == 1
    assert evaluation.confusion_matrix["global"]["__missing__"] == 1
    assert evaluation.confusion_matrix["__extra__"]["global"] == 1
