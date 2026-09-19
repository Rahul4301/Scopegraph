from collections.abc import Callable

import pytest

from scopegraph.backends.flat_graph import FlatGraphMemory
from scopegraph.backends.two_level_graph import TwoLevelGraphMemory
from scopegraph.backends.vector_memory import VectorMemory
from scopegraph.graph.in_memory import InMemoryMemoryRepository
from scopegraph.memory.base import MemorySystem
from scopegraph.memory.traversal import MemoryNeighbor
from scopegraph.models.memory import MemoryCandidate, MemoryStatus, MemoryType, ScopeLevel
from scopegraph.models.scope import ScopeCreate, ScopeRef, ScopeType
from scopegraph.models.session import SessionInput
from scopegraph.models.source import MessageRole, SourceMessage, SourceMessageCreate


class BaselineEmbeddingProvider:
    model_name = "phase4-keyword-v1"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        groups = (
            ("database", " db ", "neo4j", "mongodb", "sqlite", "postgresql"),
            ("alpha",),
            ("beta",),
            ("neo4j",),
            ("mongodb",),
            ("sqlite",),
            ("postgresql",),
            ("temporary", "testing", "right now"),
        )
        return [
            [
                1.0
                if any(term in f" {text.casefold()} " for term in group)
                else 0.0
                for group in groups
            ]
            for text in texts
        ]


class MessageExtractor:
    async def extract(
        self,
        messages: list[SourceMessage],
        *,
        current_scope: ScopeRef | None,
        existing_memories: list[str] | None = None,
    ) -> list[MemoryCandidate]:
        del current_scope, existing_memories
        candidates: list[MemoryCandidate] = []
        for message in messages:
            temporary = "temporary" in message.content.casefold()
            parts = message.content.removesuffix(".").split()
            candidates.append(
                MemoryCandidate(
                    content=message.content,
                    memory_type=(
                        MemoryType.TASK_STATE if temporary else MemoryType.DECISION
                    ),
                    subject=parts[0],
                    predicate="uses_database",
                    object=parts[-1],
                    proposed_scope_level="session" if temporary else "scope",
                    confidence=1.0,
                    durability=0.1 if temporary else 1.0,
                    source_message_ids=[message.id],
                )
            )
        return candidates


class TrackingRepository(InMemoryMemoryRepository):
    def __init__(self) -> None:
        super().__init__()
        self.neighbor_calls = 0

    async def get_memory_neighbors(
        self, memory_ids: list[str]
    ) -> list[MemoryNeighbor]:
        self.neighbor_calls += 1
        return await super().get_memory_neighbors(memory_ids)


BackendFactory = Callable[
    [InMemoryMemoryRepository, MessageExtractor, BaselineEmbeddingProvider], MemorySystem
]


async def seed_scopes(repository: InMemoryMemoryRepository) -> None:
    await repository.create_scope(
        ScopeCreate(id="global", name="Global", scope_type=ScopeType.GLOBAL)
    )
    for scope_id, name in (("alpha", "Alpha"), ("beta", "Beta")):
        await repository.create_scope(
            ScopeCreate(
                id=scope_id,
                name=name,
                scope_type=ScopeType.PROJECT,
                parent_scope_id="global",
            )
        )


async def ingest(
    system: MemorySystem,
    *,
    session_id: str,
    scope_id: str,
    content: str,
) -> None:
    await system.ingest_session(
        SessionInput(
            id=session_id,
            scope_id=scope_id,
            messages=[
                SourceMessageCreate(
                    id=f"message-{session_id}",
                    session_id=session_id,
                    role=MessageRole.USER,
                    content=content,
                    turn_index=0,
                )
            ],
        ),
        current_scope=ScopeRef(id=scope_id, name=scope_id.title(), session_id=session_id),
    )


@pytest.mark.parametrize(
    ("backend_type", "backend_name"),
    [
        (VectorMemory, "vector_memory"),
        (FlatGraphMemory, "flat_graph"),
        (TwoLevelGraphMemory, "two_level_graph"),
    ],
)
@pytest.mark.asyncio
async def test_baselines_share_interface_and_expose_cross_scope_memory(
    backend_type: BackendFactory, backend_name: str
) -> None:
    repository = InMemoryMemoryRepository()
    await seed_scopes(repository)
    system = backend_type(repository, MessageExtractor(), BaselineEmbeddingProvider())
    assert isinstance(system, MemorySystem)
    await ingest(
        system,
        session_id="alpha-session",
        scope_id="alpha",
        content="Alpha uses Neo4j",
    )
    await ingest(
        system,
        session_id="beta-session",
        scope_id="beta",
        content="Beta uses MongoDB",
    )

    result = await system.retrieve(
        "What database does Beta use?",
        current_scope=ScopeRef(id="beta", name="Beta", session_id="beta-session"),
        top_k=10,
        token_budget=100,
    )

    assert result.backend_name == backend_name
    assert {item.content for item in result.items} == {
        "Alpha uses Neo4j",
        "Beta uses MongoDB",
    }
    assert all(step.final_score is not None for step in result.trace)
    assert (await system.stats()).backend_name == backend_name


@pytest.mark.asyncio
async def test_vector_skips_graph_expansion_while_flat_graph_uses_it() -> None:
    vector_repository = TrackingRepository()
    flat_repository = TrackingRepository()
    await seed_scopes(vector_repository)
    await seed_scopes(flat_repository)
    vector = VectorMemory(
        vector_repository, MessageExtractor(), BaselineEmbeddingProvider()
    )
    flat = FlatGraphMemory(flat_repository, MessageExtractor(), BaselineEmbeddingProvider())
    for system in (vector, flat):
        await ingest(
            system,
            session_id=f"{system.kind.value}-session",
            scope_id="alpha",
            content="Alpha uses Neo4j",
        )
        await system.retrieve(
            "What database is used?",
            current_scope=ScopeRef(id="alpha"),
            top_k=4,
            token_budget=100,
        )

    assert vector_repository.neighbor_calls == 0
    assert flat_repository.neighbor_calls > 0


@pytest.mark.asyncio
async def test_two_level_keeps_only_current_session_and_global_memory() -> None:
    repository = InMemoryMemoryRepository()
    await seed_scopes(repository)
    system = TwoLevelGraphMemory(
        repository, MessageExtractor(), BaselineEmbeddingProvider()
    )
    await ingest(
        system,
        session_id="beta-durable",
        scope_id="beta",
        content="Beta uses MongoDB",
    )
    await ingest(
        system,
        session_id="beta-temporary",
        scope_id="beta",
        content="Temporary testing SQLite",
    )

    matching = await system.retrieve(
        "What are we testing right now?",
        current_scope=ScopeRef(id="beta", session_id="beta-temporary"),
        top_k=10,
        token_budget=100,
    )
    other_session = await system.retrieve(
        "What are we testing right now?",
        current_scope=ScopeRef(id="beta", session_id="new-session"),
        top_k=10,
        token_budget=100,
    )

    matching_by_content = {item.content: item for item in matching.items}
    assert matching_by_content["Temporary testing SQLite"].scope_level is ScopeLevel.SESSION
    assert matching_by_content["Beta uses MongoDB"].scope_level is ScopeLevel.GLOBAL
    assert "Temporary testing SQLite" not in {item.content for item in other_session.items}
    assert "Beta uses MongoDB" in {item.content for item in other_session.items}


@pytest.mark.parametrize(
    ("backend_type", "expected_active"),
    [
        (VectorMemory, 2),
        (FlatGraphMemory, 1),
        (TwoLevelGraphMemory, 1),
    ],
)
@pytest.mark.asyncio
async def test_conflict_domain_matches_baseline_representation(
    backend_type: BackendFactory, expected_active: int
) -> None:
    repository = InMemoryMemoryRepository()
    await seed_scopes(repository)
    system = backend_type(repository, MessageExtractor(), BaselineEmbeddingProvider())
    await ingest(
        system,
        session_id="alpha-conflict",
        scope_id="alpha",
        content="Application uses MongoDB",
    )
    await ingest(
        system,
        session_id="beta-conflict",
        scope_id="beta",
        content="Application uses PostgreSQL",
    )

    memories = await repository.list_memories(include_inactive=True)
    active = [memory for memory in memories if memory.status is MemoryStatus.ACTIVE]
    assert len(active) == expected_active
    if backend_type is VectorMemory:
        assert repository.supersedes == set()
    else:
        assert len(repository.supersedes) == 1
