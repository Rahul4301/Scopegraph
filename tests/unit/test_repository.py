import pytest

from scopegraph.graph.in_memory import InMemoryMemoryRepository
from scopegraph.models.memory import (
    MemoryCreate,
    MemoryStatus,
    MemoryType,
    MemoryUpdate,
    ScopeLevel,
)
from scopegraph.models.scope import ScopeCreate, ScopeType, ScopeUpdate
from scopegraph.models.session import SessionCreate
from scopegraph.models.source import MessageRole, SourceMessageCreate


@pytest.fixture
def repository() -> InMemoryMemoryRepository:
    return InMemoryMemoryRepository()


@pytest.mark.asyncio
async def test_scope_session_source_and_memory_crud(
    repository: InMemoryMemoryRepository,
) -> None:
    global_scope = await repository.create_scope(
        ScopeCreate(id="global", name="Global", scope_type=ScopeType.GLOBAL)
    )
    alpha = await repository.create_scope(
        ScopeCreate(
            id="alpha",
            name="Alpha",
            scope_type=ScopeType.PROJECT,
            parent_scope_id=global_scope.id,
        )
    )
    session = await repository.create_session(SessionCreate(id="session-1", scope_id=alpha.id))
    source = await repository.create_source_message(
        SourceMessageCreate(
            id="message-1",
            session_id=session.id,
            role=MessageRole.USER,
            content="Alpha uses Neo4j.",
            turn_index=0,
        )
    )
    memory = await repository.create_memory(
        MemoryCreate(
            id="memory-1",
            content="Alpha uses Neo4j",
            memory_type=MemoryType.DECISION,
            scope_level=ScopeLevel.SCOPE,
            scope_id=alpha.id,
            source_ids=[source.id],
        )
    )

    assert (await repository.get_scope(alpha.id)) == alpha
    assert (await repository.get_session(session.id)) == session
    assert (await repository.get_source_message(source.id)) == source
    assert (await repository.get_memory(memory.id)) == memory
    assert await repository.list_memories(scope_id=alpha.id) == [memory]


@pytest.mark.asyncio
async def test_memory_update_increments_revision_and_inactive_filter(
    repository: InMemoryMemoryRepository,
) -> None:
    await repository.create_scope(
        ScopeCreate(id="global", name="Global", scope_type=ScopeType.GLOBAL)
    )
    memory = await repository.create_memory(
        MemoryCreate(
            id="memory-1",
            content="Incorrect memory",
            memory_type=MemoryType.FACT,
            scope_level=ScopeLevel.GLOBAL,
            scope_id="global",
        )
    )
    updated = await repository.update_memory(
        memory.id, MemoryUpdate(status=MemoryStatus.TOMBSTONED)
    )

    assert updated is not None
    assert updated.revision == 2
    assert updated.status is MemoryStatus.TOMBSTONED
    assert await repository.list_memories() == []
    assert await repository.list_memories(include_inactive=True) == [updated]


@pytest.mark.asyncio
async def test_exactly_one_active_global_root(repository: InMemoryMemoryRepository) -> None:
    await repository.create_scope(
        ScopeCreate(id="global-1", name="Global", scope_type=ScopeType.GLOBAL)
    )
    with pytest.raises(ValueError, match="global root"):
        await repository.create_scope(
            ScopeCreate(id="global-2", name="Another", scope_type=ScopeType.GLOBAL)
        )


@pytest.mark.asyncio
async def test_archived_scope_hidden_by_default(repository: InMemoryMemoryRepository) -> None:
    await repository.create_scope(
        ScopeCreate(id="global", name="Global", scope_type=ScopeType.GLOBAL)
    )
    await repository.update_scope("global", ScopeUpdate(archived=True))
    assert await repository.list_scopes() == []
    assert len(await repository.list_scopes(include_archived=True)) == 1

