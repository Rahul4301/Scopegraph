import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evals.adapters import (
    LoCoMoAdapter,
    LongMemEvalAdapter,
)
from evals.adapters.external import evidence_source_ids, session_inputs
from evals.analysis.aggregate import aggregate_records, score_record
from evals.judges.official import LOCOMO_JUDGE_MODEL, OfficialBenchmarkJudge
from evals.metrics.official import official_score, supplemental_official_scores
from evals.runners.run_external import _terminal_summary, run_external
from evals.schemas import EvaluationRecord, ExternalBenchmarkExample
from scopegraph.memory.retriever import ScopeAwareRetriever


def _write_json(tmp_path, name: str, payload: object):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_longmemeval_adapter_normalizes_sessions(tmp_path) -> None:
    path = _write_json(
        tmp_path,
        "longmemeval.json",
        [
            {
                "question_id": "q1",
                "question_type": "single-session",
                "question": "Which database?",
                "answer": "Neo4j",
                "question_date": "2025-01-03T00:00:00Z",
                "haystack_session_ids": ["s1"],
                "haystack_dates": ["2025-01-01"],
                "haystack_sessions": [
                    [
                        {"role": "user", "content": "We chose Neo4j."},
                    ]
                ],
                "answer_session_ids": ["s1"],
            }
        ],
    )
    adapter = LongMemEvalAdapter()
    example = adapter.load(path)[0]
    assert adapter.validate(path).model_dump()["turn_count"] == 1
    assert session_inputs(example)[0].messages[0].content == "We chose Neo4j."
    inputs = session_inputs(example, as_of=example.question_date)
    assert evidence_source_ids(example, inputs) == ["q1:s1:0"]


def test_locomo_adapter_preserves_evidence_turn_ids(tmp_path) -> None:
    path = _write_json(
        tmp_path,
        "locomo.json",
        [
            {
                "sample_id": "conversation-1",
                "conversation": {
                    "session_1_date_time": "2024-01-01",
                    "session_1": [{"speaker": "A", "dia_id": "D1", "text": "Hello"}],
                },
                "qa": [
                    {
                        "question": "What happened?",
                        "answer": "Hello",
                        "category": "1",
                        "evidence": ["D1"],
                    }
                ],
            }
        ],
    )
    example = LoCoMoAdapter().load(path)[0]
    assert len(example.sessions) == 1
    assert example.metadata["evidence_turn_ids"] == ["D1"]


def test_locomo_adapter_keeps_speaker_and_shared_image(tmp_path) -> None:
    path = _write_json(
        tmp_path,
        "locomo.json",
        [{
            "sample_id": "conversation-1",
            "conversation": {
                "session_1_date_time": "2024-01-01",
                "session_1": [
                    {"speaker": "Caroline", "dia_id": "D1:1",
                     "text": "I went to a support group."},
                    {"speaker": "Melanie", "dia_id": "D1:2", "text": "Look at this!",
                     "blip_caption": "a photo of a pottery bowl"},
                ],
            },
            "qa": [{"question": "Who went?", "answer": "Caroline", "category": "4"}],
        }],
    )
    example = LoCoMoAdapter().load(path)[0]
    contents = [message.content for message in session_inputs(example)[0].messages]
    assert contents == [
        "Caroline: I went to a support group.",
        "Melanie: Look at this! [shared an image: a photo of a pottery bowl]",
    ]
    assert evidence_source_ids(
        example.model_copy(update={"metadata": {"evidence_turn_ids": ["D1:2"]}}),
        session_inputs(example),
    ) == ["conversation-1:q0:D1:2"]


def test_locomo_adapter_parses_release_session_dates(tmp_path) -> None:
    path = _write_json(
        tmp_path,
        "locomo.json",
        [{
            "sample_id": "conversation-1",
            "conversation": {
                "session_1_date_time": "1:56 pm on 8 May, 2023",
                "session_1": [{"speaker": "A", "dia_id": "D1", "text": "Yesterday"}],
                "session_2_date_time": "2:20 pm on 10 May, 2023",
                "session_2": [{"speaker": "A", "dia_id": "D2", "text": "Today"}],
            },
            "qa": [{"question": "When?", "answer": "7 May 2023", "category": "2"}],
        }],
    )
    example = LoCoMoAdapter().load(path)[0]
    assert example.sessions[0].date == datetime(2023, 5, 8, 13, 56, tzinfo=UTC)
    assert session_inputs(example)[0].messages[0].timestamp == example.sessions[0].date
    assert example.question_date == datetime(2023, 5, 10, 14, 20, tzinfo=UTC)
    assert len(session_inputs(example, as_of=example.question_date)) == 2


def _locomo_question(category: str, answer: str) -> ExternalBenchmarkExample:
    return ExternalBenchmarkExample(
        dataset="locomo", example_id="c:q0", question_id="c:q0",
        question_type=f"category_{category}", question="When did she go?", answer=answer,
        sessions=[], metadata={"sample_id": "c", "category": category},
    )


def test_locomo_accuracy_is_judged_except_deterministic_abstention(tmp_path) -> None:
    dated = _locomo_question("2", "The week before 9 June 2023")
    assert official_score("locomo", "2 June 2023", dated, data_path=tmp_path) is None
    assert set(supplemental_official_scores("locomo", "2 June 2023", dated)) == {"locomo_f1"}
    adversarial = _locomo_question("5", "a tempting wrong answer")
    assert official_score(
        "locomo", "No information available", adversarial, data_path=tmp_path
    ) == ("llm_judge_accuracy", 1.0)


@pytest.mark.asyncio
async def test_locomo_judge_grades_meaning_with_pinned_model() -> None:
    calls: list[dict] = []

    class FakeTransport:
        last_usage = {"prompt_tokens": 50, "completion_tokens": 1}

        async def post(self, url, *, api_key, payload):
            calls.append(payload)
            return {"choices": [{"message": {"content": "Same week.\nVERDICT: CORRECT"}}]}

    judge = OfficialBenchmarkJudge(base_url="https://example.test/v1", api_key="test")
    judge.transport = FakeTransport()  # type: ignore[assignment]
    result = await judge.judge(
        "locomo", _locomo_question("2", "The week before 9 June 2023"), "2 June 2023"
    )
    assert result is not None and result.score == 1.0
    assert result.metric == "llm_judge_accuracy" and result.model == LOCOMO_JUDGE_MODEL
    assert calls[0]["model"] == LOCOMO_JUDGE_MODEL and calls[0]["temperature"] == 0
    assert "The week before 9 June 2023" in calls[0]["messages"][0]["content"]
    assert await judge.judge("locomo", _locomo_question("5", "x"), "anything") is None


@pytest.mark.asyncio
async def test_locomo_runner_rejects_pre_date_fix_extraction_cache(tmp_path) -> None:
    path = _write_json(
        tmp_path,
        "locomo.json",
        [{
            "sample_id": "conversation-1",
            "conversation": {
                "session_1_date_time": "1:56 pm on 8 May, 2023",
                "session_1": [{"speaker": "A", "dia_id": "D1", "text": "Hello"}],
            },
            "qa": [{"question": "What?", "answer": "Hello", "category": "1"}],
        }],
    )
    cache = _write_json(
        tmp_path,
        "old-extractions.json",
        {
            "metadata": {
                "dataset": "locomo",
                "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "llm_model": None,
            },
            "corpora": {},
        },
    )
    with pytest.raises(ValueError, match="adapter version"):
        await run_external(
            dataset="locomo",
            path=path,
            system_name="scopegraph",
            output=tmp_path / "new.jsonl",
            storage="memory",
            extraction_cache=cache,
        )


def test_external_validation_reports_missing_file(tmp_path) -> None:
    result = LongMemEvalAdapter().validate(tmp_path / "missing.json")
    assert result.valid is False
    assert result.errors


@pytest.mark.asyncio
async def test_external_runner_replays_local_release(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path,
        "longmemeval.json",
        [
            {
                "question_id": "q1",
                "question": "Which database?",
                "answer": "Neo4j",
                "haystack_sessions": [[{"role": "user", "content": "We use Neo4j."}]],
            }
        ],
    )
    output = tmp_path / "records.jsonl"
    created = await run_external(
        dataset="longmemeval",
        path=path,
        system_name="scopegraph",
        output=output,
        storage="memory",
    )
    record = json.loads(created.read_text())
    assert record["dataset"] == "longmemeval"
    assert record["retrieved_memory_ids"]
    assert record["answer"] is None
    assert record["answer_evaluated"] is False


@pytest.mark.asyncio
async def test_external_runner_scores_all_failed_questions_as_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_json(
        tmp_path,
        "longmemeval.json",
        [
            {
                "question_id": question_id,
                "question": "Which database?",
                "answer": "Neo4j",
                "haystack_sessions": [[{"role": "user", "content": "We use Neo4j."}]],
            }
            for question_id in ("q1", "q2")
        ],
    )
    output = tmp_path / "failed.jsonl"

    async def broken_embed(self, texts):
        raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(
        "evals.runners.run_external.KeywordEmbeddingProvider.embed", broken_embed
    )
    await run_external(
        dataset="longmemeval",
        path=path,
        system_name="scopegraph",
        output=output,
        storage="memory",
    )
    records = [
        EvaluationRecord.model_validate_json(line)
        for line in output.read_text().splitlines()
    ]
    assert len(records) == 2
    assert all(record.failure_type == "RuntimeError" for record in records)
    assert all("embedding unavailable" in record.failure_message for record in records)
    assert all(record.official_score == 0.0 for record in records)
    assert all(score_record(record).metrics["official_llm_judge_accuracy"] == 0.0
               for record in records)
    summary = aggregate_records(records)["scopegraph"]
    assert summary["failure_count"] == 2.0
    assert summary["official_llm_judge_accuracy"] == 0.0
    assert "llm_judge_accuracy: 0.00% (2 questions)" in _terminal_summary(records)


def test_external_terminal_score_includes_failed_questions() -> None:
    records = [
        EvaluationRecord(
            run_id="test",
            dataset="locomo",
            system="scopegraph",
            scenario_id="conversation-1",
            question_id=f"q{index}",
            question_type="category_1",
            question="Question?",
            gold_answer="Answer",
            official_metric="locomo_f1",
            official_score=1.0 if index == 0 else 0.0,
            failure_type=None if index == 0 else "RuntimeError",
            config_hash="test",
            seed=42,
        )
        for index in range(100)
    ]
    assert aggregate_records(records)["scopegraph"]["official_locomo_f1"] == 0.01
    assert "locomo_f1: 1.00% (100 questions)" in _terminal_summary(records)


def test_external_terminal_summary_includes_secondary_official_scores() -> None:
    records = [
        EvaluationRecord(
            run_id="test",
            dataset="locomo",
            system="scopegraph",
            scenario_id="conversation-1",
            question_id="q1",
            question_type="category_2",
            question="When?",
            gold_answer="7 May 2023",
            official_metric="llm_judge_accuracy",
            official_score=1.0,
            official_secondary_scores={"locomo_f1": 0.5},
            config_hash="test",
            seed=42,
        )
    ]
    summary = _terminal_summary(records)
    assert "llm_judge_accuracy: 100.00% (1 questions)" in summary
    assert "locomo_f1: 50.00% (1 questions)" in summary


@pytest.mark.asyncio
async def test_locomo_runner_ingests_shared_history_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_json(
        tmp_path,
        "locomo.json",
        [
            {
                "sample_id": "conversation-1",
                "conversation": {
                    "session_1_date_time": "2024-01-01",
                    "session_1": [
                        {"speaker": "A", "dia_id": "D1", "text": "We use Neo4j."}
                    ],
                },
                "qa": [
                    {
                        "question": "Which database?",
                        "answer": "Neo4j",
                        "category": "1",
                        "evidence": ["D1"],
                    },
                    {
                        "question": "What do we use?",
                        "answer": "Neo4j",
                        "category": "1",
                        "evidence": ["D1"],
                    },
                ],
            },
            {
                "sample_id": "conversation-2",
                "conversation": {
                    "session_1_date_time": "2024-01-02",
                    "session_1": [
                        {"speaker": "B", "dia_id": "D2", "text": "We use SQLite."}
                    ],
                },
                "qa": [{
                    "question": "Which database?",
                    "answer": "SQLite",
                    "category": "1",
                    "evidence": ["D2"],
                }],
            },
        ],
    )
    output = tmp_path / "locomo-records.jsonl"
    observed_query_times = []
    original_retrieve = ScopeAwareRetriever.retrieve

    async def tracked_retrieve(self, query, **kwargs):
        observed_query_times.append(kwargs.get("now"))
        return await original_retrieve(self, query, **kwargs)

    monkeypatch.setattr(ScopeAwareRetriever, "retrieve", tracked_retrieve)
    created = await run_external(
        dataset="locomo",
        path=path,
        system_name="scopegraph",
        output=output,
        storage="memory",
        case_limit=1,
    )
    records = [json.loads(line) for line in created.read_text().splitlines()]
    assert len(records) == 2
    assert {record["scenario_id"] for record in records} == {"conversation-1"}
    assert records[0]["retrieved_memory_ids"] == records[1]["retrieved_memory_ids"]
    assert records[0]["gold_source_ids"] == ["conversation-1:D1"]
    assert records[0]["retrieved_source_contents"] == [["A: We use Neo4j."]]
    assert observed_query_times == [datetime(2024, 1, 1, tzinfo=UTC)] * 2


def test_external_replay_excludes_future_turns(tmp_path) -> None:
    path = _write_json(
        tmp_path,
        "longmemeval.json",
        [
            {
                "question_id": "q1",
                "question": "Which database?",
                "answer": "Neo4j",
                "question_date": "2025-01-02T00:00:00Z",
                "haystack_session_ids": ["past", "future"],
                "haystack_dates": ["2025-01-01", "2025-01-03"],
                "haystack_sessions": [
                    [{"role": "user", "content": "Neo4j"}],
                    [{"role": "user", "content": "Future leak"}],
                ],
                "answer_session_ids": ["past"],
            }
        ],
    )
    example = LongMemEvalAdapter().load(path)[0]
    inputs = session_inputs(example, as_of=example.question_date)
    assert [item.id for item in inputs] == ["q1:past"]
    assert evidence_source_ids(example, inputs) == ["q1:past:0"]

