"""Helpers for adding explicit, reproducible conflict messages."""

from scopegraph.models.source import MessageRole, SourceMessageCreate


def conflict_message(*, session_id: str, message_id: str, subject: str, old_value: str,
                     new_value: str, turn_index: int = 0) -> SourceMessageCreate:
    return SourceMessageCreate(
        id=message_id, session_id=session_id, role=MessageRole.USER,
        content=f"{subject} changed from {old_value} to {new_value}.", turn_index=turn_index,
    )
