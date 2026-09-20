"""MemConflict import adapter.

MemConflict has multiple releases and generated subsets. This adapter accepts
the stable question/session interchange shape when a local release is supplied;
it never creates synthetic data under the MemConflict name.
"""

import json
from pathlib import Path

from evals.adapters.external import load_records, normalize_turn, parse_datetime
from evals.adapters.longmemeval import _validation
from evals.schemas import AdapterValidation, ExternalBenchmarkExample, ExternalSession


class MemConflictAdapter:
    name = "memconflict"

    def load(self, path: str | Path) -> list[ExternalBenchmarkExample]:
        examples: list[ExternalBenchmarkExample] = []
        for index, raw in enumerate(load_records(path)):
            question = raw.get("question")
            answer = raw.get("answer", raw.get("gold_answer"))
            if not question or answer is None:
                raise ValueError(
                    f"MemConflict record {index} requires question and answer/gold_answer"
                )
            raw_sessions = raw.get("sessions", raw.get("haystack_sessions", []))
            sessions = []
            for session_index, session in enumerate(raw_sessions):
                if isinstance(session, list):
                    turns = session
                    session_id = str(session_index)
                    date = None
                elif isinstance(session, dict):
                    session_id = str(session.get("session_id", session.get("id", session_index)))
                    turns = session.get("turns", session.get("messages", []))
                    date = parse_datetime(session.get("date", session.get("timestamp")))
                else:
                    raise ValueError(f"MemConflict record {index} has an invalid session")
                sessions.append(ExternalSession(
                    session_id=session_id, date=date,
                    turns=[normalize_turn(turn, fallback_id=f"{session_id}:{turn_index}")
                           for turn_index, turn in enumerate(turns)],
                ))
            example_id = str(raw.get("question_id", raw.get("id", index)))
            examples.append(ExternalBenchmarkExample(
                dataset=self.name, example_id=example_id, question_id=example_id,
                question_type=str(raw.get("question_type", raw.get("conflict_type", "unknown"))),
                question=str(question), answer=str(answer),
                question_date=parse_datetime(raw.get("question_date")), sessions=sessions,
                answer_session_ids=[str(item) for item in raw.get("answer_session_ids", [])],
                metadata={"source_fields": sorted(raw)},
            ))
        return examples

    def validate(self, path: str | Path) -> AdapterValidation:
        try:
            examples = self.load(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return AdapterValidation(dataset=self.name, path=str(path), valid=False,
                                     errors=[str(exc)])
        return _validation(self.name, path, examples)
