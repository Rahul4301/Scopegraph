"""Adapter for the public LoCoMo10 JSON release."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.adapters.external import load_records, normalize_turn, parse_datetime
from evals.adapters.longmemeval import _validation
from evals.schemas import AdapterValidation, ExternalBenchmarkExample, ExternalSession


class LoCoMoAdapter:
    name = "locomo"

    def load(self, path: str | Path) -> list[ExternalBenchmarkExample]:
        examples: list[ExternalBenchmarkExample] = []
        for sample_index, raw in enumerate(load_records(path)):
            conversation = raw.get("conversation")
            if not isinstance(conversation, dict):
                raise ValueError(f"LoCoMo sample {sample_index} has no conversation object")
            sessions = _sessions(conversation)
            # LoCoMo asks questions after the full conversation, but does not
            # provide per-question timestamps. Use the last observed turn instead
            # of letting retrieval treat old memories as if queried today.
            question_date = max(
                timestamp
                for session in sessions
                for timestamp in [session.date, *(turn.timestamp for turn in session.turns)]
                if timestamp is not None
            )
            sample_id = str(raw.get("sample_id", f"sample-{sample_index}"))
            for question_index, question in enumerate(raw.get("qa", [])):
                if not isinstance(question, dict) or "question" not in question:
                    raise ValueError(f"LoCoMo sample {sample_id} has an invalid QA record")
                category = str(question.get("category", "unknown"))
                answer = (
                    question.get("adversarial_answer")
                    if category == "5"
                    else question.get("answer")
                )
                examples.append(
                    ExternalBenchmarkExample(
                        dataset=self.name,
                        example_id=f"{sample_id}:q{question_index}",
                        question_id=f"{sample_id}:q{question_index}",
                        question_type=f"category_{category}",
                        question=str(question["question"]),
                        answer=str(answer or ""),
                        question_date=question_date,
                        sessions=sessions,
                        metadata={
                            "sample_id": sample_id,
                            "category": category,
                            "evidence_turn_ids": [
                                str(item) for item in question.get("evidence", [])
                            ],
                        },
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


def _sessions(conversation: dict[str, Any]) -> list[ExternalSession]:
    sessions: list[ExternalSession] = []
    session_numbers = sorted({
        int(key.removeprefix("session_").removesuffix("_date_time"))
        for key in conversation
        if _session_number(key) is not None
    })
    for number in session_numbers:
        turns = conversation.get(f"session_{number}", [])
        if not isinstance(turns, list):
            raise ValueError(f"LoCoMo session_{number} must be a list")
        raw_date = conversation.get(f"session_{number}_date_time")
        date = parse_datetime(raw_date)
        if date is None:
            try:
                date = datetime.strptime(
                    str(raw_date), "%I:%M %p on %d %B, %Y"
                ).replace(tzinfo=UTC)
            except ValueError as exc:
                raise ValueError(
                    f"LoCoMo session_{number} has an invalid date: {raw_date!r}"
                ) from exc
        sessions.append(ExternalSession(
            session_id=str(number), date=date,
            turns=[normalize_turn(_attributed_turn(turn), fallback_id=f"{number}:{index}")
                   for index, turn in enumerate(turns)],
        ))
    if not sessions:
        raise ValueError("LoCoMo conversation contains no session_N arrays")
    return sessions


def _attributed_turn(turn: Any) -> Any:
    """Keep who said each turn and what image they shared.

    LoCoMo is a dialogue between two named people, so ``speaker`` is identity rather
    than a chat role; role normalization would reduce both speakers to ``user`` and
    leave first-person turns unattributable. Shared photos exist only as captions.
    """
    if not isinstance(turn, dict):
        return turn
    speaker = str(turn.get("speaker") or "").strip()
    text = str(turn.get("text") or "").strip()
    caption = str(turn.get("blip_caption") or "").strip()
    content = f"{speaker}: {text}" if speaker else text
    if caption:
        content = f"{content} [shared an image: {caption}]"
    return {**turn, "text": content}


def _session_number(key: str) -> int | None:
    if not key.startswith("session_"):
        return None
    suffix = key.removeprefix("session_").removesuffix("_date_time")
    return int(suffix) if suffix.isdigit() else None
