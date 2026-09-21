"""Shared validity rules applied before embedding, traversal, and ranking."""

from datetime import datetime

from scopegraph.models.memory import Memory, MemoryStatus, ScopeLevel


def eligible_memory(memory: Memory, *, scope_ids: set[str], session_id: str | None,
                    historical: bool, now: datetime) -> bool:
    statuses = {MemoryStatus.ACTIVE}
    if historical:
        statuses |= {MemoryStatus.SUPERSEDED, MemoryStatus.ARCHIVED}
    return (
        memory.scope_id in scope_ids
        and memory.status in statuses
        and (memory.valid_from is None or memory.valid_from <= now)
        and (historical or memory.valid_to is None or memory.valid_to > now)
        and (memory.scope_level is not ScopeLevel.SESSION or (
            session_id is not None and memory.metadata.get("session_id") == session_id
        ))
    )
