import json
from pathlib import Path

import pytest

from evals.adapters import (
    LoCoMoAdapter,
    LongMemEvalAdapter,
    LongMemEvalV2Adapter,
    Mem2ActBenchAdapter,
    MemBenchAdapter,
    MemConflictAdapter,
    MemoryAgentBenchAdapter,
    RHELMAdapter,
    TIMEAdapter,
)
from evals.adapters.external import session_inputs
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
    assert example.metadata["evidence_turn_ids"] == ["D1"]


def test_memconflict_adapter_accepts_local_interchange(tmp_path) -> None:
    path = _write_json(
        tmp_path,
        "memconflict.json",
        [
            {
                "id": "m1",
                "question": "What is the current plan?",
                "gold_answer": "Plan B",
                "conflict_type": "dynamic",
                "sessions": [{"id": "s1", "turns": [{"role": "user", "text": "Plan B"}]}],
            }
        ],
    )
    result = MemConflictAdapter().validate(path)
    assert result.valid is True
    assert result.example_count == 1


def test_external_validation_reports_missing_file(tmp_path) -> None:
    result = LongMemEvalAdapter().validate(tmp_path / "missing.json")
    assert result.valid is False
    assert result.errors


@pytest.mark.parametrize(
    "adapter",
    [
        LongMemEvalV2Adapter(),
        MemoryAgentBenchAdapter(),
        RHELMAdapter(),
        MemBenchAdapter(),
        Mem2ActBenchAdapter(),
        TIMEAdapter(),
    ],
)
def test_extended_adapters_normalize_release_interchange(tmp_path, adapter) -> None:
    path = _write_json(
        tmp_path,
        f"{adapter.name}.json",
        [
            {
                "id": "q1",
                "query": "Which database is current?",
                "target": "Neo4j",
                "category": "knowledge_update",
                "question_date": "2026-01-03",
                "supporting_evidence": ["session-2:0"],
                "history": [
                    {
                        "id": "session-1",
                        "date": "2026-01-01",
                        "messages": [{"role": "user", "text": "We used Postgres."}],
                    },
                    {
                        "id": "session-2",
                        "date": "2026-01-02",
                        "messages": [{"role": "user", "text": "Now we use Neo4j."}],
                    },
                ],
            }
        ],
    )
    example = adapter.load(path)[0]
    assert adapter.validate(path).valid is True
    assert example.answer == "Neo4j"
    assert example.answer_session_ids == ["session-2:0"]
    assert len(example.sessions) == 2


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
        dataset="longmemeval", path=path, system_name="scopegraph", output=output
    )
    record = json.loads(created.read_text())
    assert record["dataset"] == "longmemeval"
    assert record["retrieved_memory_ids"]
