from scopegraph.memory.normalizer import conflict_key, normalize_text
from scopegraph.models.memory import Memory, MemoryCandidate, MemoryStatus, ScopeLevel


def find_conflicts(
    candidate: MemoryCandidate, existing: list[Memory], *, scope_id: str
) -> list[Memory]:
    key = conflict_key(candidate)
    if key is None or candidate.object is None:
        return []
    conflicts: list[Memory] = []
    if candidate.proposed_scope_level == ScopeLevel.SESSION.value:
        return conflicts
    for memory in existing:
        if memory.scope_id != scope_id or memory.status is not MemoryStatus.ACTIVE:
            continue
        if memory.scope_level is ScopeLevel.SESSION:
            continue
        if memory.metadata.get("conflict_key") != key:
            continue
        previous_object = memory.metadata.get("object")
        if isinstance(previous_object, str) and normalize_text(previous_object) != candidate.object:
            conflicts.append(memory)
    return conflicts
