"""Adapters for newer memory benchmarks with evolving public release layouts.

These adapters preserve the source fields and accept a small, explicit set of
aliases used by the official releases. They never synthesize missing histories.
"""

import json
from pathlib import Path
from typing import Any

from evals.adapters.external import load_records, normalize_turn, parse_datetime
from evals.adapters.longmemeval import _validation
from evals.schemas import AdapterValidation, ExternalBenchmarkExample, ExternalSession


class FlexibleMemoryBenchmarkAdapter:
    name = "external"
    history_fields = (
        "sessions",
        "haystack_sessions",
        "history",
        "conversation",
        "conversations",
        "context",
        "timeline",
        "events",
        "episodes",
    )

    def load(self, path: str | Path) -> list[ExternalBenchmarkExample]:
        examples: list[ExternalBenchmarkExample] = []
        for index, raw in enumerate(load_records(path)):
            question = _first(raw, "question", "query", "prompt", "input")
            answer = _first(raw, "answer", "gold_answer", "target", "reference_answer")
            if question is None or answer is None:
                raise ValueError(
                    f"{self.name} record {index} requires a question/query and answer/target"
                )
            history = next((raw[field] for field in self.history_fields if field in raw), None)
            if history is None:
                raise ValueError(
                    f"{self.name} record {index} has no history; expected one of "
                    f"{list(self.history_fields)}"
                )
            sessions = _sessions(history, record_index=index)
            if not sessions:
                raise ValueError(f"{self.name} record {index} contains no usable sessions")
            example_id = str(_first(raw, "question_id", "id", "example_id") or index)
            evidence = _list_of_strings(
                _first(
                    raw,
                    "answer_session_ids",
                    "supporting_evidence",
                    "evidence",
                    "gold_evidence",
                )
            )
            metadata = {
                "source_fields": sorted(raw),
                "evidence": evidence,
                "characteristics": _list_of_strings(raw.get("characteristics")),
            }
            examples.append(
                ExternalBenchmarkExample(
                    dataset=self.name,
                    example_id=example_id,
                    question_id=example_id,
                    question_type=str(
                        _first(raw, "question_type", "task_type", "category", "type") or "unknown"
                    ),
                    question=str(question),
                    answer=str(answer),
                    question_date=parse_datetime(
                        _first(raw, "question_date", "query_date", "timestamp", "date")
                    ),
                    sessions=sessions,
                    answer_session_ids=evidence,
                    metadata=metadata,
                )
            )
        return examples

    def validate(self, path: str | Path) -> AdapterValidation:
        try:
            examples = self.load(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return AdapterValidation(
                dataset=self.name, path=str(path), valid=False, errors=[str(exc)]
            )
        return _validation(self.name, path, examples)


class LongMemEvalV2Adapter(FlexibleMemoryBenchmarkAdapter):
    name = "longmemeval_v2"


class MemoryAgentBenchAdapter(FlexibleMemoryBenchmarkAdapter):
    name = "memoryagentbench"


class RHELMAdapter(FlexibleMemoryBenchmarkAdapter):
    name = "rhelm"


class MemBenchAdapter(FlexibleMemoryBenchmarkAdapter):
    name = "membench"


class Mem2ActBenchAdapter(FlexibleMemoryBenchmarkAdapter):
    name = "mem2actbench"


class TIMEAdapter(FlexibleMemoryBenchmarkAdapter):
    name = "time"


def _first(raw: dict[str, Any], *keys: str) -> Any:
    return next((raw[key] for key in keys if key in raw and raw[key] is not None), None)


def _list_of_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _sessions(raw: Any, *, record_index: int) -> list[ExternalSession]:
    if isinstance(raw, str):
        raw = [{"id": "0", "turns": [{"role": "user", "content": raw}]}]
    elif isinstance(raw, dict):
        if _looks_like_turn(raw):
            raw = [[raw]]
        else:
            raw = [
                {"id": key, "turns": value}
                for key, value in sorted(raw.items(), key=lambda item: str(item[0]))
            ]
    if not isinstance(raw, list):
        raise ValueError(f"record {record_index} history must be a list, object, or string")
    if raw and all(isinstance(item, dict) and _looks_like_turn(item) for item in raw):
        raw = [raw]

    sessions: list[ExternalSession] = []
    for session_index, session in enumerate(raw):
        session_id = str(session_index)
        date = None
        if isinstance(session, str):
            turns: Any = [{"role": "user", "content": session}]
        elif isinstance(session, list):
            turns = session
        elif isinstance(session, dict):
            session_id = str(
                _first(session, "session_id", "id", "date", "timestamp") or session_index
            )
            date = parse_datetime(_first(session, "date", "timestamp", "created_at"))
            turns = _first(
                session,
                "turns",
                "messages",
                "dialogue",
                "conversation",
                "events",
                "content",
                "text",
            )
            if isinstance(turns, str):
                turns = [{"role": "user", "content": turns}]
        else:
            raise ValueError(f"record {record_index} contains an invalid session")
        if not isinstance(turns, list):
            raise ValueError(f"record {record_index} session {session_id} has no turn list")
        normalized = []
        for turn_index, turn in enumerate(turns):
            if isinstance(turn, str):
                turn = {"role": "user", "content": turn}
            if not isinstance(turn, dict):
                raise ValueError(f"record {record_index} session {session_id} has an invalid turn")
            normalized.append(normalize_turn(turn, fallback_id=f"{session_id}:{turn_index}"))
        sessions.append(ExternalSession(session_id=session_id, date=date, turns=normalized))
    return sessions


def _looks_like_turn(raw: dict[str, Any]) -> bool:
    return any(field in raw for field in ("content", "text", "utterance"))
