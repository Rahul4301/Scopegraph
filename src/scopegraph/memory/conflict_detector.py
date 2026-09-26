import re

from scopegraph.memory.normalizer import conflict_key, normalize_text
from scopegraph.models.memory import (
    Memory,
    MemoryCandidate,
    MemoryStatus,
    MemoryType,
    ScopeLevel,
)

# Events and relationships accumulate: attending a second event or adopting a
# second pet adds a value rather than replacing the first.
ACCUMULATING_TYPES = {
    MemoryType.EVENT,
    MemoryType.RELATIONSHIP,
    MemoryType.SUMMARY,
    MemoryType.OTHER,
}
# Attribute slots that hold one current value, matching the predicate vocabulary the
# extraction prompt requests (uses_database, preferred_language, ...). A trailing
# preposition marks a purpose or relation (uses_art_for) rather than a slot.
SINGLE_VALUED_PREDICATE = re.compile(
    r"^(uses|preferred|chosen|current|primary)_[a-z0-9_]+(?<!_for)(?<!_with)(?<!_to)"
    r"(?<!_by)(?<!_about)$"
)


def find_conflicts(
    candidate: MemoryCandidate, existing: list[Memory], *, scope_id: str
) -> list[Memory]:
    key = conflict_key(candidate)
    if key is None or candidate.object is None:
        return []
    conflicts: list[Memory] = []
    if candidate.proposed_scope_level == ScopeLevel.SESSION.value:
        return conflicts
    if not _replaces_previous_value(candidate):
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


def _replaces_previous_value(candidate: MemoryCandidate) -> bool:
    """Whether a new object for the same subject|predicate supersedes the old one.

    Returning False keeps both values active (e.g. two pets, two events attended).
    """
    if candidate.possible_contradiction:
        return True
    if candidate.memory_type in ACCUMULATING_TYPES or not candidate.predicate:
        return False
    predicate = normalize_text(candidate.predicate).replace(" ", "_")
    return SINGLE_VALUED_PREDICATE.fullmatch(predicate) is not None
