import json

from evals.adapters import LoCoMoAdapter, LongMemEvalAdapter, MemConflictAdapter
from evals.adapters.external import session_inputs


def _write_json(tmp_path, name: str, payload: object):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_longmemeval_adapter_normalizes_sessions(tmp_path) -> None:
    path = _write_json(
        tmp_path,
        "longmemeval.json",
        [{
            "question_id": "q1",
            "question_type": "single-session",
            "question": "Which database?",
            "answer": "Neo4j",
            "question_date": "2025-01-03T00:00:00Z",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2025-01-01"],
            "haystack_sessions": [[
                {"role": "user", "content": "We chose Neo4j."},
            ]],
            "answer_session_ids": ["s1"],
        }],
    )
    adapter = LongMemEvalAdapter()
    example = adapter.load(path)[0]
    assert adapter.validate(path).model_dump()["turn_count"] == 1
    assert session_inputs(example)[0].messages[0].content == "We chose Neo4j."


def test_locomo_adapter_preserves_evidence_turn_ids(tmp_path) -> None:
    path = _write_json(
        tmp_path,
        "locomo.json",
        [{
            "sample_id": "conversation-1",
            "conversation": {
                "session_1_date_time": "2024-01-01",
                "session_1": [{"speaker": "A", "dia_id": "D1", "text": "Hello"}],
            },
            "qa": [{
                "question": "What happened?", "answer": "Hello",
                "category": "1", "evidence": ["D1"],
            }],
        }],
    )
    example = LoCoMoAdapter().load(path)[0]
    assert example.metadata["evidence_turn_ids"] == ["D1"]


def test_memconflict_adapter_accepts_local_interchange(tmp_path) -> None:
    path = _write_json(
        tmp_path,
        "memconflict.json",
        [{
            "id": "m1",
            "question": "What is the current plan?",
            "gold_answer": "Plan B",
            "conflict_type": "dynamic",
            "sessions": [{"id": "s1", "turns": [{"role": "user", "text": "Plan B"}]}],
        }],
    )
    result = MemConflictAdapter().validate(path)
    assert result.valid is True
    assert result.example_count == 1


def test_external_validation_reports_missing_file(tmp_path) -> None:
    result = LongMemEvalAdapter().validate(tmp_path / "missing.json")
    assert result.valid is False
    assert result.errors
