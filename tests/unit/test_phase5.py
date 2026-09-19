import pytest

from scopegraph.graph.in_memory import InMemoryMemoryRepository
from scopegraph.memory.corrections import CorrectionService
from scopegraph.models.correction import (
    CorrectionAction,
    CorrectionRelation,
    MemoryEditRequest,
    MemoryMergeRequest,
    MemoryMoveRequest,
    MemoryRestoreRequest,
    MemorySupersedeRequest,
    RelationCorrectionRequest,
)
from scopegraph.models.memory import (
    Memory,
    MemoryCreate,
    MemoryStatus,
    MemoryType,
    ScopeLevel,
)
from scopegraph.models.relationship import RelationKind
from scopegraph.models.scope import ScopeCreate, ScopeType
from scopegraph.models.session import SessionCreate
from scopegraph.models.source import MessageRole, SourceMessageCreate


async def correction_fixture() -> tuple[
    InMemoryMemoryRepository, CorrectionService, dict[str, Memory]
]:
    repository = InMemoryMemoryRepository()
    await repository.create_scope(
        ScopeCreate(id="global", name="Global", scope_type=ScopeType.GLOBAL)
    )
    for scope_id in ("alpha", "beta"):
        await repository.create_scope(
            ScopeCreate(
                id=scope_id,
                name=scope_id.title(),
                scope_type=ScopeType.PROJECT,
                parent_scope_id="global",
            )
        )
    memories: dict[str, Memory] = {}
    for memory_id in ("bad", "good", "sole-dependent", "shared-dependent"):
        memories[memory_id] = await repository.create_memory(
            MemoryCreate(
                id=memory_id,
                content=f"Memory {memory_id}",
                memory_type=MemoryType.FACT,
                scope_level=ScopeLevel.SCOPE,
                scope_id="alpha",
            )
        )
    await repository.link_support("bad", "sole-dependent")
    await repository.link_support("bad", "shared-dependent")
    await repository.link_support("good", "shared-dependent")
    return repository, CorrectionService(repository), memories


@pytest.mark.asyncio
async def test_edit_creates_revision_event_and_invalidates_embedding() -> None:
    repository, service, memories = await correction_fixture()
    await repository.set_memory_embedding("bad", [1.0, 0.0], "old-model")

    result = await service.edit(
        "bad",
        MemoryEditRequest(
            content="Corrected memory",
            actor="researcher",
            reason="source was wrong",
        ),
    )

    assert result.memory.content == "Corrected memory"
    assert result.memory.revision == memories["bad"].revision + 1
    assert result.memory.embedding is None
    assert result.memory.embedding_model is None
    assert result.event.action is CorrectionAction.EDIT
    assert result.event.before["memory"]["content"] == "Memory bad"
    assert result.event.after["memory"]["content"] == "Corrected memory"
    assert await service.history("bad") == [result.event]


@pytest.mark.asyncio
async def test_move_archive_and_restore_are_audited() -> None:
    _, service, _ = await correction_fixture()
    moved = await service.move(
        "bad",
        MemoryMoveRequest(
            scope_id="beta", actor="researcher", reason="belongs to Beta"
        ),
    )
    assert moved.memory.scope_id == "beta"
    assert moved.memory.scope_level is ScopeLevel.SCOPE

    archived = await service.archive("bad", actor="researcher", reason="not current")
    assert archived.memory.status is MemoryStatus.ARCHIVED
    restored = await service.restore(
        "bad",
        MemoryRestoreRequest(
            undo_of=archived.event.id,
            actor="researcher",
            reason="needed again",
        ),
    )
    assert restored.memory.status is MemoryStatus.ACTIVE
    assert restored.event.undo_of == archived.event.id
    assert [event.action for event in await service.history("bad")] == [
        CorrectionAction.MOVE_SCOPE,
        CorrectionAction.ARCHIVE,
        CorrectionAction.RESTORE,
    ]


@pytest.mark.asyncio
async def test_prune_preview_distinguishes_evidentiary_dependencies() -> None:
    _, service, _ = await correction_fixture()

    preview = await service.preview_prune("bad")
    impacts = {impact.memory_id: impact for impact in preview.dependencies}

    assert preview.hard_delete is False
    assert impacts["sole-dependent"].proposed_status is MemoryStatus.NEEDS_REVIEW
    assert impacts["sole-dependent"].other_active_support_ids == []
    assert impacts["shared-dependent"].proposed_status is MemoryStatus.ACTIVE
    assert impacts["shared-dependent"].other_active_support_ids == ["good"]
    assert all(neighbor.evidentiary_dependency for neighbor in preview.graph_neighbors)


@pytest.mark.asyncio
async def test_prune_and_undo_only_change_unsupported_dependents() -> None:
    repository, service, _ = await correction_fixture()

    pruned = await service.prune("bad", actor="researcher", reason="bad evidence")
    assert pruned.memory.status is MemoryStatus.TOMBSTONED
    assert [memory.id for memory in pruned.affected_memories] == ["sole-dependent"]
    assert (await repository.get_memory("sole-dependent")).status is MemoryStatus.NEEDS_REVIEW  # type: ignore[union-attr]
    assert (await repository.get_memory("shared-dependent")).status is MemoryStatus.ACTIVE  # type: ignore[union-attr]

    restored = await service.restore(
        "bad",
        MemoryRestoreRequest(undo_of=pruned.event.id, reason="undo prune"),
    )
    assert restored.memory.status is MemoryStatus.ACTIVE
    assert [memory.id for memory in restored.affected_memories] == ["sole-dependent"]
    assert (await repository.get_memory("sole-dependent")).status is MemoryStatus.ACTIVE  # type: ignore[union-attr]
    assert [event.action for event in await service.history("sole-dependent")] == [
        CorrectionAction.TOMBSTONE,
        CorrectionAction.RESTORE,
    ]


@pytest.mark.asyncio
async def test_restore_refuses_to_overwrite_a_newer_revision() -> None:
    _, service, _ = await correction_fixture()
    pruned = await service.prune("bad")
    await service.edit("bad", MemoryEditRequest(content="Edited after prune"))

    with pytest.raises(ValueError, match="changed after"):
        await service.restore(
            "bad", MemoryRestoreRequest(undo_of=pruned.event.id)
        )


@pytest.mark.asyncio
async def test_merge_preserves_provenance_and_tombstones_duplicate() -> None:
    repository = InMemoryMemoryRepository()
    await repository.create_scope(
        ScopeCreate(id="global", name="Global", scope_type=ScopeType.GLOBAL)
    )
    await repository.create_session(SessionCreate(id="session", scope_id="global"))
    for index in (1, 2):
        await repository.create_source_message(
            SourceMessageCreate(
                id=f"message-{index}",
                session_id="session",
                role=MessageRole.USER,
                content=f"Evidence {index}",
                turn_index=index,
            )
        )
    target = await repository.create_memory(
        MemoryCreate(
            id="target",
            content="Canonical",
            memory_type=MemoryType.FACT,
            scope_level=ScopeLevel.GLOBAL,
            scope_id="global",
            source_ids=["message-1"],
        )
    )
    source = await repository.create_memory(
        MemoryCreate(
            id="source",
            content="Duplicate",
            memory_type=MemoryType.FACT,
            scope_level=ScopeLevel.GLOBAL,
            scope_id="global",
            source_ids=["message-2"],
        )
    )
    service = CorrectionService(repository)

    result = await service.merge(
        source.id,
        MemoryMergeRequest(
            target_memory_id=target.id,
            actor="researcher",
            reason="same fact",
        ),
    )

    assert result.memory.id == "target"
    assert result.memory.source_ids == ["message-1", "message-2"]
    assert result.affected_memories[0].status is MemoryStatus.TOMBSTONED
    assert ("source", "target") in repository.same_as
    assert (await service.history("source"))[0].action is CorrectionAction.MERGE
    assert (await service.history("target"))[0].action is CorrectionAction.MERGE


@pytest.mark.asyncio
async def test_supersede_and_allowlisted_relation_changes_are_audited() -> None:
    repository, service, _ = await correction_fixture()
    superseded = await service.supersede(
        "bad",
        MemorySupersedeRequest(
            replacement_memory_id="good",
            reason="newer fact",
        ),
    )
    assert superseded.affected_memories[0].status is MemoryStatus.SUPERSEDED
    assert ("good", "bad") in repository.supersedes
    assert ("good", "bad") in repository.contradicts

    relation = RelationCorrectionRequest(
        target_memory_id="shared-dependent",
        relation=CorrectionRelation.RELATES_TO,
        kind=RelationKind.USES,
        reason="explicit semantic relation",
    )
    added = await service.add_relation("good", relation)
    assert ("good", "shared-dependent", RelationKind.USES) in repository.relates_to
    removed = await service.remove_relation("good", relation)
    assert ("good", "shared-dependent", RelationKind.USES) not in repository.relates_to
    assert added.event.action is CorrectionAction.ADD_RELATION
    assert removed.event.action is CorrectionAction.REMOVE_RELATION
