"""LongMemEval cleaned/oracle JSON adapter.

The dataset is intentionally not bundled. Download it from the official
repository/Hugging Face release, then point this adapter at the local file.
"""

import json
from pathlib import Path

from evals.adapters.external import load_records, normalize_turn, parse_datetime
from evals.schemas import AdapterValidation, ExternalBenchmarkExample, ExternalSession


class LongMemEvalAdapter:
    name = "longmemeval"

    def load(self, path: str | Path) -> list[ExternalBenchmarkExample]:
        examples: list[ExternalBenchmarkExample] = []
        for index, raw in enumerate(load_records(path)):
            required = ("question_id", "question", "answer", "haystack_sessions")
            missing = [field for field in required if field not in raw]
            if missing:
                raise ValueError(f"LongMemEval record {index} missing fields: {missing}")
            raw_haystack = raw["haystack_sessions"]
            if not isinstance(raw_haystack, list):
                raise ValueError(f"LongMemEval record {index} haystack_sessions must be a list")
            session_ids = [str(item) for item in raw.get("haystack_session_ids", [])]
            dates = list(raw.get("haystack_dates", []))
            sessions = []
            for session_index, turns in enumerate(raw_haystack):
                if not isinstance(turns, list):
                    raise ValueError(f"LongMemEval record {index} session must be a list")
                session_id = (
                    session_ids[session_index]
                    if session_index < len(session_ids)
                    else str(session_index)
                )
                date = parse_datetime(dates[session_index]) if session_index < len(dates) else None
                normalized_turns = [
                    normalize_turn(turn, fallback_id=f"{session_id}:{turn_index}")
                    for turn_index, turn in enumerate(turns)
                ]
                sessions.append(
                    ExternalSession(
                        session_id=session_id,
                        date=date,
                        turns=normalized_turns,
                    )
                )
            examples.append(
                ExternalBenchmarkExample(
                    dataset=self.name,
                    example_id=str(raw["question_id"]),
                    question_id=str(raw["question_id"]),
                    question_type=str(raw.get("question_type", "unknown")),
                    question=str(raw["question"]),
                    answer=str(raw["answer"]),
                    question_date=parse_datetime(raw.get("question_date")),
                    sessions=sessions,
                    answer_session_ids=[str(item) for item in raw.get("answer_session_ids", [])],
                    metadata={"source_fields": sorted(raw)},
                )
            )
        return examples

    def validate(self, path: str | Path) -> AdapterValidation:
        try:
            examples = self.load(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return AdapterValidation(dataset=self.name, path=str(path), valid=False,
                                     errors=[str(exc)])
        return _validation(self.name, path, examples)


def _validation(dataset: str, path: str | Path,
                examples: list[ExternalBenchmarkExample]) -> AdapterValidation:
    sessions = sum(len(example.sessions) for example in examples)
    turns = sum(len(session.turns) for example in examples for session in example.sessions)
    return AdapterValidation(dataset=dataset, path=str(path), valid=True,
                             example_count=len(examples), session_count=sessions, turn_count=turns)
