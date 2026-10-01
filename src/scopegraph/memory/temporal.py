from datetime import datetime

from scopegraph.models.memory import Memory, MemoryStatus


def supersession_time(new_memory: Memory) -> datetime:
    """Time the new memory takes effect, which ends the superseded memory's validity."""
    return new_memory.valid_from or new_memory.created_at


def is_historical_query(query: str, historical_terms: tuple[str, ...]) -> bool:
    """Whether a query asks about past state."""
    lowered = query.casefold()
    return any(term.casefold() in lowered for term in historical_terms)


def temporal_score(memory: Memory, *, now: datetime, historical: bool) -> float:
    """Temporal validity score of a memory at ``now``."""
    if memory.status is MemoryStatus.TOMBSTONED:
        return 0.0
    if memory.valid_from and memory.valid_from > now:
        return 0.0
    if historical:
        if memory.status is MemoryStatus.SUPERSEDED:
            return 1.0
        if memory.status is MemoryStatus.ACTIVE:
            return 0.7
        return 0.4
    if memory.status is not MemoryStatus.ACTIVE:
        return 0.0
    if memory.valid_from and memory.valid_from > now:
        return 0.0
    if memory.valid_to and memory.valid_to <= now:
        return 0.0
    return 1.0
