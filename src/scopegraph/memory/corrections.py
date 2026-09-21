from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from functools import wraps
from typing import Any, Concatenate, ParamSpec, Protocol

from scopegraph.memory.traversal import MemoryNeighbor
from scopegraph.models.correction import (
    CorrectionAction,
    CorrectionEvent,
    CorrectionRelation,
    CorrectionRequest,
    CorrectionResult,
    GraphNeighborPreview,
    MemoryEditRequest,
    MemoryMergeRequest,
    MemoryMoveRequest,
    MemoryRestoreRequest,
    MemorySupersedeRequest,
    PruneImpact,
    PrunePreview,
    RelationCorrectionRequest,
    SupportDependency,
)
from scopegraph.models.memory import Memory, MemoryStatus, MemoryUpdate, ScopeLevel
from scopegraph.models.relationship import RelationKind
from scopegraph.models.scope import Scope, ScopeType


class CorrectionRepository(Protocol):
    def transaction(self) -> AbstractAsyncContextManager[None]: ...

    async def get_memory(self, memory_id: str) -> Memory | None: ...

    async def get_scope(self, scope_id: str) -> Scope | None: ...

    async def update_memory(self, memory_id: str, update: MemoryUpdate) -> Memory | None: ...

    async def move_memory(
        self, memory_id: str, scope_id: str, scope_level: ScopeLevel
    ) -> Memory | None: ...

    async def merge_memories(
        self, source_memory_id: str, target_memory_id: str
    ) -> tuple[Memory, Memory]: ...

    async def supersede_memory(self, old_memory_id: str, new_memory_id: str) -> None: ...

    async def add_memory_relation(
        self,
        source_memory_id: str,
        target_memory_id: str,
        relation: CorrectionRelation,
        kind: RelationKind | None = None,
    ) -> Memory: ...

    async def remove_memory_relation(
        self,
        source_memory_id: str,
        target_memory_id: str,
        relation: CorrectionRelation,
        kind: RelationKind | None = None,
    ) -> Memory: ...

    async def get_memory_neighbors(self, memory_ids: list[str]) -> list[MemoryNeighbor]: ...

    async def get_support_dependents(self, memory_id: str) -> list[SupportDependency]: ...

    async def create_correction_event(
        self, event: CorrectionEvent, target_memory_ids: list[str]
    ) -> CorrectionEvent: ...

    async def get_correction_event(self, event_id: str) -> CorrectionEvent | None: ...

    async def list_correction_events(self, memory_id: str) -> list[CorrectionEvent]: ...


def _snapshot(memory: Memory) -> dict[str, Any]:
    return memory.model_dump(mode="json")


P = ParamSpec("P")


def atomic(
    method: Callable[Concatenate["CorrectionService", P], Awaitable[CorrectionResult]],
) -> Callable[Concatenate["CorrectionService", P], Awaitable[CorrectionResult]]:
    @wraps(method)
    async def wrapped(
        self: "CorrectionService", /, *args: P.args, **kwargs: P.kwargs
    ) -> CorrectionResult:
        async with self.repository.transaction():
            return await method(self, *args, **kwargs)

    return wrapped


class CorrectionService:
    def __init__(self, repository: CorrectionRepository) -> None:
        self.repository = repository

    async def history(self, memory_id: str) -> list[CorrectionEvent]:
        await self._require_memory(memory_id)
        return await self.repository.list_correction_events(memory_id)

    @atomic
    async def edit(self, memory_id: str, request: MemoryEditRequest) -> CorrectionResult:
        before = await self._require_memory(memory_id)
        changes = request.model_dump(exclude={"actor", "reason"}, exclude_unset=True)
        if not changes:
            raise ValueError("At least one editable memory field is required")
        if "content" in changes:
            changes.update(embedding=None, embedding_model=None)
            # Structured extraction describes the old content and must not cause
            # a future duplicate or conflict decision after a human text edit.
            changes["metadata"] = {
                key: value
                for key, value in before.metadata.items()
                if key not in {"subject", "predicate", "object", "normalized_key", "conflict_key"}
            }
        Memory.model_validate(
            {
                **before.model_dump(),
                **changes,
                "revision": before.revision + 1,
            }
        )
        updated = await self.repository.update_memory(
            memory_id, MemoryUpdate.model_validate(changes)
        )
        if updated is None:
            raise ValueError(f"Memory {memory_id!r} does not exist")
        event = await self._record(
            CorrectionAction.EDIT,
            before={"memory": _snapshot(before)},
            after={"memory": _snapshot(updated)},
            actor=request.actor,
            reason=request.reason,
            target_ids=[memory_id],
        )
        return CorrectionResult(event=event, memory=updated)

    @atomic
    async def move(self, memory_id: str, request: MemoryMoveRequest) -> CorrectionResult:
        before = await self._require_memory(memory_id)
        scope = await self.repository.get_scope(request.scope_id)
        if scope is None:
            raise ValueError(f"Scope {request.scope_id!r} does not exist")
        inferred_level = (
            ScopeLevel.GLOBAL if scope.scope_type is ScopeType.GLOBAL else ScopeLevel.SCOPE
        )
        scope_level = request.scope_level or inferred_level
        if (scope.scope_type is ScopeType.GLOBAL) != (scope_level is ScopeLevel.GLOBAL):
            raise ValueError("Global scope and global memory level must be used together")
        updated = await self.repository.move_memory(memory_id, scope.id, scope_level)
        if updated is None:
            raise ValueError(f"Memory {memory_id!r} does not exist")
        event = await self._record(
            CorrectionAction.MOVE_SCOPE,
            before={"memory": _snapshot(before)},
            after={"memory": _snapshot(updated)},
            actor=request.actor,
            reason=request.reason,
            target_ids=[memory_id],
        )
        return CorrectionResult(event=event, memory=updated)

    @atomic
    async def archive(
        self, memory_id: str, *, actor: str = "user", reason: str = ""
    ) -> CorrectionResult:
        before = await self._require_memory(memory_id)
        if before.status in {MemoryStatus.ARCHIVED, MemoryStatus.TOMBSTONED}:
            raise ValueError("Archived or tombstoned memory cannot be archived again")
        updated = await self.repository.update_memory(
            memory_id, MemoryUpdate(status=MemoryStatus.ARCHIVED)
        )
        if updated is None:
            raise ValueError(f"Memory {memory_id!r} does not exist")
        event = await self._record(
            CorrectionAction.ARCHIVE,
            before={"memory": _snapshot(before)},
            after={"memory": _snapshot(updated)},
            actor=actor,
            reason=reason,
            target_ids=[memory_id],
        )
        return CorrectionResult(event=event, memory=updated)

    async def preview_prune(self, memory_id: str) -> PrunePreview:
        memory = await self._require_memory(memory_id)
        dependencies = await self.repository.get_support_dependents(memory_id)
        impacts = [self._impact(dependency) for dependency in dependencies]
        dependency_ids = {impact.memory_id for impact in impacts}
        neighbors = await self.repository.get_memory_neighbors([memory_id])
        graph_neighbors = [
            GraphNeighborPreview(
                memory_id=neighbor.memory.id,
                relation=neighbor.relation,
                evidentiary_dependency=neighbor.memory.id in dependency_ids,
            )
            for neighbor in neighbors
        ]
        return PrunePreview(
            memory_id=memory.id,
            current_status=memory.status,
            dependencies=impacts,
            graph_neighbors=graph_neighbors,
        )

    @atomic
    async def prune(
        self, memory_id: str, *, actor: str = "user", reason: str = ""
    ) -> CorrectionResult:
        before = await self._require_memory(memory_id)
        if before.status is MemoryStatus.TOMBSTONED:
            raise ValueError("Memory is already tombstoned")
        preview = await self.preview_prune(memory_id)
        dependent_before: list[Memory] = []
        dependent_after: list[Memory] = []
        for impact in preview.dependencies:
            if impact.proposed_status is not MemoryStatus.NEEDS_REVIEW:
                continue
            dependent = await self._require_memory(impact.memory_id)
            updated = await self.repository.update_memory(
                dependent.id, MemoryUpdate(status=MemoryStatus.NEEDS_REVIEW)
            )
            if updated is None:
                raise ValueError(f"Memory {dependent.id!r} does not exist")
            dependent_before.append(dependent)
            dependent_after.append(updated)
        updated_target = await self.repository.update_memory(
            memory_id, MemoryUpdate(status=MemoryStatus.TOMBSTONED)
        )
        if updated_target is None:
            raise ValueError(f"Memory {memory_id!r} does not exist")
        event = await self._record(
            CorrectionAction.TOMBSTONE,
            before={
                "memory": _snapshot(before),
                "dependents": [_snapshot(memory) for memory in dependent_before],
            },
            after={
                "memory": _snapshot(updated_target),
                "dependents": [_snapshot(memory) for memory in dependent_after],
            },
            actor=actor,
            reason=reason,
            target_ids=[memory_id, *[memory.id for memory in dependent_after]],
        )
        return CorrectionResult(
            event=event,
            memory=updated_target,
            affected_memories=dependent_after,
        )

    @atomic
    async def restore(self, memory_id: str, request: MemoryRestoreRequest) -> CorrectionResult:
        before = await self._require_memory(memory_id)
        if before.status not in {MemoryStatus.TOMBSTONED, MemoryStatus.ARCHIVED}:
            raise ValueError("Only archived or tombstoned memories can be restored")
        undo_event = await self._resolve_undo_event(memory_id, request.undo_of)
        if undo_event is None:
            raise ValueError("Memory has no restorable archive or tombstone event")
        event_after = undo_event.after.get("memory", {})
        if before.revision != event_after.get("revision"):
            raise ValueError("Memory changed after the correction and cannot be safely restored")
        previous_status = MemoryStatus.ACTIVE
        previous = undo_event.before.get("memory", {})
        if "status" in previous:
            previous_status = MemoryStatus(previous["status"])
        restored = await self.repository.update_memory(
            memory_id, MemoryUpdate(status=previous_status)
        )
        if restored is None:
            raise ValueError(f"Memory {memory_id!r} does not exist")

        dependent_before: list[Memory] = []
        dependent_after: list[Memory] = []
        if undo_event.action is CorrectionAction.TOMBSTONE:
            before_by_id = {item["id"]: item for item in undo_event.before.get("dependents", [])}
            for after_snapshot in undo_event.after.get("dependents", []):
                current = await self.repository.get_memory(after_snapshot["id"])
                original = before_by_id.get(after_snapshot["id"])
                if current is None or original is None:
                    continue
                if (
                    current.status is not MemoryStatus.NEEDS_REVIEW
                    or current.revision != after_snapshot.get("revision")
                ):
                    continue
                updated = await self.repository.update_memory(
                    current.id, MemoryUpdate(status=MemoryStatus(original["status"]))
                )
                if updated is not None:
                    dependent_before.append(current)
                    dependent_after.append(updated)
        event = await self._record(
            CorrectionAction.RESTORE,
            before={
                "memory": _snapshot(before),
                "dependents": [_snapshot(memory) for memory in dependent_before],
            },
            after={
                "memory": _snapshot(restored),
                "dependents": [_snapshot(memory) for memory in dependent_after],
            },
            actor=request.actor,
            reason=request.reason,
            target_ids=[memory_id, *[memory.id for memory in dependent_after]],
            undo_of=undo_event.id,
        )
        return CorrectionResult(
            event=event,
            memory=restored,
            affected_memories=dependent_after,
        )

    @atomic
    async def merge(self, memory_id: str, request: MemoryMergeRequest) -> CorrectionResult:
        if memory_id == request.target_memory_id:
            raise ValueError("A memory cannot be merged into itself")
        source_before = await self._require_memory(memory_id)
        target_before = await self._require_memory(request.target_memory_id)
        if source_before.status is not MemoryStatus.ACTIVE:
            raise ValueError("Merge source must be active")
        if target_before.status is not MemoryStatus.ACTIVE:
            raise ValueError("Merge target must be active")
        source_after, target_after = await self.repository.merge_memories(
            memory_id, request.target_memory_id
        )
        event = await self._record(
            CorrectionAction.MERGE,
            before={
                "source": _snapshot(source_before),
                "target": _snapshot(target_before),
            },
            after={
                "source": _snapshot(source_after),
                "target": _snapshot(target_after),
            },
            actor=request.actor,
            reason=request.reason,
            target_ids=[memory_id, request.target_memory_id],
        )
        return CorrectionResult(
            event=event,
            memory=target_after,
            affected_memories=[source_after],
        )

    @atomic
    async def supersede(self, memory_id: str, request: MemorySupersedeRequest) -> CorrectionResult:
        if memory_id == request.replacement_memory_id:
            raise ValueError("A memory cannot supersede itself")
        old_before = await self._require_memory(memory_id)
        new_before = await self._require_memory(request.replacement_memory_id)
        if old_before.status is not MemoryStatus.ACTIVE:
            raise ValueError("Superseded memory must currently be active")
        if new_before.status is not MemoryStatus.ACTIVE:
            raise ValueError("Replacement memory must currently be active")
        await self.repository.supersede_memory(memory_id, request.replacement_memory_id)
        old_after = await self._require_memory(memory_id)
        new_after = await self._require_memory(request.replacement_memory_id)
        event = await self._record(
            CorrectionAction.SUPERSEDE,
            before={"old": _snapshot(old_before), "new": _snapshot(new_before)},
            after={"old": _snapshot(old_after), "new": _snapshot(new_after)},
            actor=request.actor,
            reason=request.reason,
            target_ids=[memory_id, request.replacement_memory_id],
        )
        return CorrectionResult(
            event=event,
            memory=new_after,
            affected_memories=[old_after],
        )

    @atomic
    async def add_relation(
        self, memory_id: str, request: RelationCorrectionRequest
    ) -> CorrectionResult:
        source_before = await self._require_memory(memory_id)
        target = await self._require_memory(request.target_memory_id)
        source_after = await self.repository.add_memory_relation(
            memory_id,
            target.id,
            request.relation,
            request.kind,
        )
        event = await self._record(
            CorrectionAction.ADD_RELATION,
            before={
                "source": _snapshot(source_before),
                "target": _snapshot(target),
                "relation_present": False,
            },
            after={
                "source": _snapshot(source_after),
                "target": _snapshot(target),
                "relation_present": True,
                "relation": request.relation.value,
                "kind": request.kind.value if request.kind else None,
            },
            actor=request.actor,
            reason=request.reason,
            target_ids=[memory_id, target.id],
        )
        return CorrectionResult(event=event, memory=source_after)

    @atomic
    async def remove_relation(
        self, memory_id: str, request: RelationCorrectionRequest
    ) -> CorrectionResult:
        source_before = await self._require_memory(memory_id)
        target = await self._require_memory(request.target_memory_id)
        source_after = await self.repository.remove_memory_relation(
            memory_id,
            target.id,
            request.relation,
            request.kind,
        )
        event = await self._record(
            CorrectionAction.REMOVE_RELATION,
            before={
                "source": _snapshot(source_before),
                "target": _snapshot(target),
                "relation_present": True,
                "relation": request.relation.value,
                "kind": request.kind.value if request.kind else None,
            },
            after={
                "source": _snapshot(source_after),
                "target": _snapshot(target),
                "relation_present": False,
            },
            actor=request.actor,
            reason=request.reason,
            target_ids=[memory_id, target.id],
        )
        return CorrectionResult(event=event, memory=source_after)

    async def apply(self, correction: CorrectionRequest) -> CorrectionResult:
        context: dict[str, Any] = {
            "actor": correction.actor,
            "reason": correction.reason,
        }
        if correction.action is CorrectionAction.EDIT:
            return await self.edit(
                correction.memory_id,
                MemoryEditRequest.model_validate({**context, **correction.changes}),
            )
        if correction.action is CorrectionAction.MOVE_SCOPE:
            return await self.move(
                correction.memory_id,
                MemoryMoveRequest.model_validate({**context, **correction.changes}),
            )
        if correction.action is CorrectionAction.ARCHIVE:
            return await self.archive(correction.memory_id, **context)
        if correction.action is CorrectionAction.TOMBSTONE:
            return await self.prune(correction.memory_id, **context)
        if correction.action is CorrectionAction.RESTORE:
            return await self.restore(
                correction.memory_id,
                MemoryRestoreRequest.model_validate({**context, "undo_of": correction.undo_of}),
            )
        if correction.action is CorrectionAction.MERGE:
            return await self.merge(
                correction.memory_id,
                MemoryMergeRequest.model_validate({**context, **correction.changes}),
            )
        if correction.action is CorrectionAction.SUPERSEDE:
            return await self.supersede(
                correction.memory_id,
                MemorySupersedeRequest.model_validate({**context, **correction.changes}),
            )
        if correction.action is CorrectionAction.ADD_RELATION:
            return await self.add_relation(
                correction.memory_id,
                RelationCorrectionRequest.model_validate({**context, **correction.changes}),
            )
        if correction.action is CorrectionAction.REMOVE_RELATION:
            return await self.remove_relation(
                correction.memory_id,
                RelationCorrectionRequest.model_validate({**context, **correction.changes}),
            )
        raise ValueError(f"Unsupported correction action: {correction.action}")

    async def _require_memory(self, memory_id: str) -> Memory:
        memory = await self.repository.get_memory(memory_id)
        if memory is None:
            raise ValueError(f"Memory {memory_id!r} does not exist")
        return memory

    async def _resolve_undo_event(
        self, memory_id: str, event_id: str | None
    ) -> CorrectionEvent | None:
        if event_id is not None:
            event = await self.repository.get_correction_event(event_id)
            if event is None:
                raise ValueError(f"Correction event {event_id!r} does not exist")
            if event.action not in {CorrectionAction.TOMBSTONE, CorrectionAction.ARCHIVE}:
                raise ValueError("Only tombstone or archive events can be restored")
            event_memory = event.before.get("memory", {})
            if event_memory.get("id") != memory_id:
                raise ValueError("Correction event does not target this memory as its subject")
            return event
        history = await self.repository.list_correction_events(memory_id)
        return next(
            (
                event
                for event in reversed(history)
                if event.action in {CorrectionAction.TOMBSTONE, CorrectionAction.ARCHIVE}
            ),
            None,
        )

    @staticmethod
    def _impact(dependency: SupportDependency) -> PruneImpact:
        memory = dependency.memory
        should_review = (
            memory.status is MemoryStatus.ACTIVE and not dependency.other_active_support_ids
        )
        return PruneImpact(
            memory_id=memory.id,
            current_status=memory.status,
            proposed_status=(MemoryStatus.NEEDS_REVIEW if should_review else memory.status),
            other_active_support_ids=dependency.other_active_support_ids,
            reason=(
                "target is the only active evidentiary support"
                if should_review
                else "independent active support remains or dependent is already inactive"
            ),
        )

    async def _record(
        self,
        action: CorrectionAction,
        *,
        before: dict[str, Any],
        after: dict[str, Any],
        actor: str,
        reason: str,
        target_ids: list[str],
        undo_of: str | None = None,
    ) -> CorrectionEvent:
        event = CorrectionEvent(
            action=action,
            actor=actor,
            reason=reason,
            before=before,
            after=after,
            undo_of=undo_of,
        )
        return await self.repository.create_correction_event(event, target_ids)
