"""Shared safe parsing helpers for downloaded external benchmark files."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.schemas import ExternalBenchmarkExample, ExternalTurn
from scopegraph.models.session import SessionInput
from scopegraph.models.source import MessageRole, SourceMessageCreate


def load_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {source}")
    if source.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    else:
        payload = json.loads(source.read_text())
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            candidate = payload.get("data", payload.get("examples", payload))
            records = candidate if isinstance(candidate, list) else [candidate]
        else:
            raise ValueError("Dataset JSON must contain an object or list")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every dataset record must be a JSON object")
    return records


def parse_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def normalize_role(value: Any) -> str:
    role = str(value or "user").casefold()
    return role if role in {item.value for item in MessageRole} else "user"


def normalize_turn(raw: dict[str, Any], *, fallback_id: str) -> ExternalTurn:
    content = raw.get("content", raw.get("text", raw.get("utterance", "")))
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Turn {fallback_id} has no text content")
    return ExternalTurn(
        role=normalize_role(raw.get("role", raw.get("speaker"))), content=content,
        timestamp=parse_datetime(raw.get("timestamp", raw.get("date"))),
        turn_id=str(raw.get("id", raw.get("dia_id", fallback_id))),
        has_answer=bool(raw.get("has_answer", False)),
    )


def session_inputs(
    example: ExternalBenchmarkExample,
    *,
    scope_id: str | None = None,
    as_of: datetime | None = None,
) -> list[SessionInput]:
    """Convert normalized sessions into the common MemorySystem ingestion model."""
    target_scope = scope_id or f"external_{example.dataset}_{example.example_id}"
    inputs: list[SessionInput] = []
    ordered_sessions = sorted(
        enumerate(example.sessions),
        key=lambda item: (
            item[1].date or example.question_date or datetime.min.replace(tzinfo=UTC),
            item[0],
        ),
    )
    for _, session in ordered_sessions:
        started = session.date or example.question_date or datetime.now(UTC)
        messages = [
            SourceMessageCreate(
                id=f"{example.example_id}:{turn.turn_id or f'{session.session_id}:{index}'}",
                session_id=f"{example.example_id}:{session.session_id}",
                role=MessageRole(turn.role),
                content=turn.content,
                timestamp=turn.timestamp or started, turn_index=index,
            )
            for index, turn in enumerate(session.turns)
            if as_of is None or (turn.timestamp or started) <= as_of
        ]
        if (as_of is not None and started > as_of) or not messages:
            continue
        inputs.append(
            SessionInput(
                id=f"{example.example_id}:{session.session_id}",
                scope_id=target_scope,
                started_at=started,
                messages=messages,
            )
        )
    return inputs


def evidence_source_ids(
    example: ExternalBenchmarkExample, inputs: list[SessionInput]
) -> list[str]:
    """Resolve release evidence identifiers to normalized source-message IDs.

    Releases variously identify an entire session or an individual turn. Unknown
    identifiers remain absent instead of being guessed, making missing gold visible
    as missing rather than granting accidental retrieval credit.
    """
    evidence = {
        *example.answer_session_ids,
        *(str(item) for item in example.metadata.get("evidence_turn_ids", [])),
    }
    resolved: set[str] = set()
    for normalized in inputs:
        for identifier in evidence:
            normalized_id = f"{example.example_id}:{identifier}"
            if normalized.id == normalized_id:
                resolved.update(message.id for message in normalized.messages)
            resolved.update(
                message.id for message in normalized.messages if message.id == normalized_id
            )
    return sorted(resolved)
