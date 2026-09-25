import json
from pathlib import Path

import pytest

from evals.adapters import (
    LoCoMoAdapter,
    LongMemEvalAdapter,
)
from evals.adapters.external import evidence_source_ids, session_inputs
from evals.runners.run_external import run_external


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
async def test_external_runner_exits_nonzero_when_question_failed(
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
    with pytest.raises(RuntimeError, match="1 of 2 questions failed"):
        await run_external(
            dataset="longmemeval",
            path=path,
            system_name="scopegraph",
            output=output,
            storage="memory",
        )
    record = json.loads(output.read_text())
    assert record["failure_type"] == "RuntimeError"
    assert "embedding unavailable" in record["failure_message"]


@pytest.mark.asyncio
async def test_locomo_runner_ingests_shared_history_once(tmp_path: Path) -> None:
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
            }
        ],
    )
    output = tmp_path / "locomo-records.jsonl"
    created = await run_external(
        dataset="locomo",
        path=path,
        system_name="scopegraph",
        output=output,
        storage="memory",
    )
    records = [json.loads(line) for line in created.read_text().splitlines()]
    assert len(records) == 2
    assert {record["scenario_id"] for record in records} == {"conversation-1"}
    assert records[0]["retrieved_memory_ids"] == records[1]["retrieved_memory_ids"]
    assert records[0]["gold_source_ids"] == ["conversation-1:D1"]


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
