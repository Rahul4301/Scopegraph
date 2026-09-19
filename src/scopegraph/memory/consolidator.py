from dataclasses import dataclass
from typing import Protocol

from scopegraph.llm.scope_classification import resolve_candidate_scope
from scopegraph.memory.conflict_detector import find_conflicts
from scopegraph.memory.deduplicator import find_duplicate
from scopegraph.memory.normalizer import candidate_key, conflict_key, normalize_candidate
from scopegraph.memory.promoter import PromotionPolicy
from scopegraph.memory.provenance import validate_provenance
from scopegraph.models.memory import Memory, MemoryCandidate, MemoryCreate, ScopeLevel
from scopegraph.models.scope import ScopeRef


class ConsolidationRepository(Protocol):
    async def list_memories(
        self, *, scope_id: str | None = None, include_inactive: bool = False
    ) -> list[Memory]: ...

    async def create_memory(self, request: MemoryCreate) -> Memory: ...

    async def supersede_memory(self, old_memory_id: str, new_memory_id: str) -> None: ...

    async def link_support(self, source_memory_id: str, target_memory_id: str) -> None: ...


@dataclass
class ConsolidationOutcome:
    memories: list[Memory]
    duplicate_count: int = 0
    conflict_count: int = 0
    promoted_count: int = 0


class Consolidator:
    def __init__(self, repository: ConsolidationRepository, policy: PromotionPolicy) -> None:
        self.repository = repository
        self.policy = policy

    async def consolidate(
        self,
        candidates: list[MemoryCandidate],
        *,
        session_id: str,
        session_message_ids: set[str],
        current_scope: ScopeRef | None,
        global_scope_id: str | None,
    ) -> ConsolidationOutcome:
        outcome = ConsolidationOutcome(memories=[])
        for raw_candidate in candidates:
            validate_provenance(raw_candidate, session_message_ids)
            candidate = normalize_candidate(raw_candidate)
            decision = resolve_candidate_scope(
                candidate, current_scope=current_scope, global_scope_id=global_scope_id
            )
            if decision.scope_id is None:
                continue
            level = ScopeLevel(decision.scope_level)
            if level is ScopeLevel.SCOPE and not self.policy.should_promote_session_to_scope(
                candidate
            ):
                level = ScopeLevel.SESSION

            existing = await self.repository.list_memories(
                scope_id=decision.scope_id, include_inactive=True
            )
            duplicate = find_duplicate(candidate, existing, scope_id=decision.scope_id)
            if duplicate is not None:
                outcome.duplicate_count += 1
                continue

            metadata = {
                "subject": candidate.subject,
                "predicate": candidate.predicate,
                "object": candidate.object,
                "normalized_key": candidate_key(candidate),
                "conflict_key": conflict_key(candidate),
                "durability": candidate.durability,
                "inferred": candidate.inferred,
                "session_id": session_id,
                "scope_reason": decision.reason,
            }
            memory = await self.repository.create_memory(
                MemoryCreate(
                    content=candidate.content,
                    memory_type=candidate.memory_type,
                    scope_level=level,
                    scope_id=decision.scope_id,
                    confidence=candidate.confidence,
                    valid_from=candidate.valid_from,
                    valid_to=candidate.valid_to,
                    source_ids=candidate.source_message_ids,
                    metadata=metadata,
                )
            )
            conflicts = find_conflicts(candidate, existing, scope_id=decision.scope_id)
            for old_memory in conflicts:
                await self.repository.supersede_memory(old_memory.id, memory.id)
            outcome.conflict_count += len(conflicts)
            outcome.memories.append(memory)

            if level is ScopeLevel.SCOPE and global_scope_id is not None:
                await self._maybe_promote(candidate, memory, global_scope_id, outcome)
        return outcome

    async def _maybe_promote(
        self,
        candidate: MemoryCandidate,
        memory: Memory,
        global_scope_id: str,
        outcome: ConsolidationOutcome,
    ) -> None:
        all_active = await self.repository.list_memories(include_inactive=False)
        key = candidate_key(candidate)
        equivalents = [
            item
            for item in all_active
            if item.scope_level is ScopeLevel.SCOPE
            and item.metadata.get("normalized_key") == key
        ]
        if not self.policy.should_promote_scope_to_global(candidate, equivalents):
            return
        global_existing = await self.repository.list_memories(
            scope_id=global_scope_id, include_inactive=False
        )
        duplicate = find_duplicate(candidate, global_existing, scope_id=global_scope_id)
        if duplicate is not None:
            await self.repository.link_support(memory.id, duplicate.id)
            return
        promoted = await self.repository.create_memory(
            MemoryCreate(
                content=candidate.content,
                memory_type=candidate.memory_type,
                scope_level=ScopeLevel.GLOBAL,
                scope_id=global_scope_id,
                confidence=candidate.confidence,
                valid_from=candidate.valid_from,
                valid_to=candidate.valid_to,
                source_ids=list(
                    dict.fromkeys(
                        source_id
                        for support in equivalents
                        for source_id in support.source_ids
                    )
                ),
                metadata={
                    **memory.metadata,
                    "promoted_from_scope_ids": sorted({item.scope_id for item in equivalents}),
                },
            )
        )
        for support in equivalents:
            await self.repository.link_support(support.id, promoted.id)
        outcome.memories.append(promoted)
        outcome.promoted_count += 1
