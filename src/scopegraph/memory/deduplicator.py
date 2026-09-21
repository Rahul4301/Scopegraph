from scopegraph.memory.normalizer import candidate_key, normalize_text
from scopegraph.models.memory import Memory, MemoryCandidate, MemoryStatus, ScopeLevel


def find_duplicate(
    candidate: MemoryCandidate, existing: list[Memory], *, scope_id: str,
    scope_level: ScopeLevel | None = None, session_id: str | None = None,
) -> Memory | None:
    key = candidate_key(candidate)
    for memory in existing:
        if memory.scope_id != scope_id or memory.status is not MemoryStatus.ACTIVE:
            continue
        if scope_level is not None and memory.scope_level is not scope_level:
            continue
        if memory.scope_level is ScopeLevel.SESSION and (
            memory.metadata.get("session_id") != session_id
        ):
            continue
        stored_key = str(memory.metadata.get("normalized_key", ""))
        if stored_key == key or normalize_text(memory.content) == normalize_text(candidate.content):
            return memory
    return None
