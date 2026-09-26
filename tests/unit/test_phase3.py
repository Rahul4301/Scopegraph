import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scopegraph.embeddings.cache import CachedEmbedder, SQLiteEmbeddingCache
from scopegraph.graph.in_memory import InMemoryMemoryRepository
from scopegraph.memory.ranker import RankingWeights, cosine_similarity, final_score
from scopegraph.memory.retriever import ScopeAwareRetriever
from scopegraph.memory.traversal import bounded_traversal
from scopegraph.models.memory import (
    MemoryCreate,
    MemoryStatus,
    MemoryType,
    ScopeLevel,
)
from scopegraph.models.retrieval import RetrievedMemory
from scopegraph.models.scope import ScopeCreate, ScopeRef, ScopeType
from scopegraph.models.session import SessionCreate
from scopegraph.models.source import MessageRole, SourceMessageCreate
from scopegraph.observability.token_counting import pack_to_token_budget


class KeywordEmbeddingProvider:
    model_name = "keyword-test-v1"

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        keywords = (
            ("database", " db ", "neo4j", "mongodb", "sqlite", "postgresql"),
            ("test", "testing", "temporarily", "right now"),
            ("prefer", "preference", "generally"),
            ("alpha",),
            ("beta",),
            ("neo4j",),
            ("mongodb",),
            ("sqlite",),
            ("postgresql", "postgres"),
        )
        vectors: list[list[float]] = []
        for text in texts:
            padded = f" {text.casefold()} "
            vectors.append(
                [1.0 if any(term in padded for term in group) else 0.0 for group in keywords]
            )
        return vectors


async def seed_retrieval_fixture(repository: InMemoryMemoryRepository) -> None:
    await repository.create_scope(
        ScopeCreate(id="global", name="Global", scope_type=ScopeType.GLOBAL)
    )
    await repository.create_scope(
        ScopeCreate(
            id="alpha",
            name="Alpha",
            scope_type=ScopeType.PROJECT,
            parent_scope_id="global",
        )
    )
    await repository.create_scope(
        ScopeCreate(
            id="beta",
            name="Beta",
            scope_type=ScopeType.PROJECT,
            parent_scope_id="global",
        )
    )
    memories = (
        MemoryCreate(
            id="global-postgres",
            content="User generally prefers PostgreSQL",
            memory_type=MemoryType.PREFERENCE,
            scope_level=ScopeLevel.GLOBAL,
            scope_id="global",
        ),
        MemoryCreate(
            id="alpha-neo4j",
            content="Alpha uses Neo4j",
            memory_type=MemoryType.DECISION,
            scope_level=ScopeLevel.SCOPE,
            scope_id="alpha",
        ),
        MemoryCreate(
            id="beta-mongodb",
            content="Beta uses MongoDB",
            memory_type=MemoryType.DECISION,
            scope_level=ScopeLevel.SCOPE,
            scope_id="beta",
        ),
        MemoryCreate(
            id="beta-sqlite",
            content="Temporarily test SQLite in Beta",
            memory_type=MemoryType.TASK_STATE,
            scope_level=ScopeLevel.SESSION,
            scope_id="beta",
            metadata={"session_id": "beta-current"},
        ),
    )
    for memory in memories:
        await repository.create_memory(memory)


@pytest.mark.asyncio
async def test_scope_aware_retrieval_fixture() -> None:
    repository = InMemoryMemoryRepository()
    await seed_retrieval_fixture(repository)
    retriever = ScopeAwareRetriever(repository, KeywordEmbeddingProvider())
    beta = ScopeRef(id="beta", name="Beta", session_id="beta-current")

    normal = await retriever.retrieve(
        "What DB does this project use normally?",
        current_scope=beta,
        top_k=4,
        token_budget=100,
    )
    assert normal.items[0].memory_id == "beta-mongodb"
    assert all(item.memory_id != "alpha-neo4j" for item in normal.items)

    testing = await retriever.retrieve(
        "What DB are we testing right now?",
        current_scope=beta,
        top_k=4,
        token_budget=100,
    )
    assert testing.items[0].memory_id == "beta-sqlite"

    global_result = await retriever.retrieve(
        "What DB do I generally prefer?",
        current_scope=None,
        top_k=4,
        token_budget=100,
    )
    assert global_result.items[0].memory_id == "global-postgres"

    named_scope = await retriever.retrieve(
        "What DB does Beta use?",
        current_scope=ScopeRef(id="alpha", name="Alpha"),
        top_k=4,
        token_budget=100,
    )
    assert named_scope.items[0].memory_id == "beta-mongodb"
    assert named_scope.trace[0].reason == "query explicitly names scope Beta"


@pytest.mark.asyncio
async def test_session_memory_does_not_leak_without_matching_session() -> None:
    repository = InMemoryMemoryRepository()
    await seed_retrieval_fixture(repository)
    retriever = ScopeAwareRetriever(repository, KeywordEmbeddingProvider())
    result = await retriever.retrieve(
        "What are we testing?",
        current_scope=ScopeRef(id="beta", session_id="different-session"),
        top_k=8,
        token_budget=100,
    )
    assert "beta-sqlite" not in {item.memory_id for item in result.items}


@pytest.mark.asyncio
async def test_historical_query_prefers_superseded_memory() -> None:
    repository = InMemoryMemoryRepository()
    await repository.create_scope(
        ScopeCreate(id="alpha", name="Alpha", scope_type=ScopeType.PROJECT)
    )
    old = await repository.create_memory(
        MemoryCreate(
            id="old",
            content="Alpha uses MongoDB",
            memory_type=MemoryType.DECISION,
            scope_level=ScopeLevel.SCOPE,
            scope_id="alpha",
        )
    )
    new = await repository.create_memory(
        MemoryCreate(
            id="new",
            content="Alpha uses Neo4j",
            memory_type=MemoryType.DECISION,
            scope_level=ScopeLevel.SCOPE,
            scope_id="alpha",
        )
    )
    await repository.supersede_memory(old.id, new.id)
    retriever = ScopeAwareRetriever(repository, KeywordEmbeddingProvider())

    current = await retriever.retrieve(
        "What database does Alpha use now?",
        current_scope=ScopeRef(id="alpha"),
        top_k=2,
        token_budget=100,
    )
    assert [item.memory_id for item in current.items] == ["new"]

    history = await retriever.retrieve(
        "What database did Alpha use before?",
        current_scope=ScopeRef(id="alpha"),
        top_k=2,
        token_budget=100,
    )
    assert history.items[0].memory_id == "old"
    assert history.items[0].status is MemoryStatus.SUPERSEDED


@pytest.mark.asyncio
async def test_retrieval_includes_budgeted_verbatim_provenance() -> None:
    repository = InMemoryMemoryRepository()
    await repository.create_scope(
        ScopeCreate(id="alpha", name="Alpha", scope_type=ScopeType.PROJECT)
    )
    occurred_at = datetime(2023, 5, 8, 13, 0, tzinfo=UTC)
    await repository.create_session(
        SessionCreate(id="session", scope_id="alpha", started_at=occurred_at)
    )
    source = await repository.create_source_message(
        SourceMessageCreate(
            id="source",
            session_id="session",
            role=MessageRole.USER,
            content="The launch was moved to Friday at 3 PM.",
            timestamp=occurred_at,
            turn_index=0,
        )
    )
    await repository.create_memory(
        MemoryCreate(
            id="launch",
            content="Launch schedule changed",
            memory_type=MemoryType.EVENT,
            scope_level=ScopeLevel.SCOPE,
            scope_id="alpha",
            source_ids=[source.id],
        )
    )
    retriever = ScopeAwareRetriever(repository, KeywordEmbeddingProvider())

    result = await retriever.retrieve(
        "When is the launch?",
        current_scope=ScopeRef(id="alpha"),
        top_k=1,
        token_budget=100,
        now=occurred_at,
    )

    assert [message.id for message in result.items[0].source_messages] == ["source"]
    assert result.items[0].source_messages[0].content == source.content
    assert result.items[0].source_messages[0].timestamp == occurred_at
    assert result.token_count <= 100


@pytest.mark.asyncio
async def test_bounded_traversal_expands_allowlisted_edges_without_cycles() -> None:
    repository = InMemoryMemoryRepository()
    await repository.create_scope(
        ScopeCreate(id="alpha", name="Alpha", scope_type=ScopeType.PROJECT)
    )
    for memory_id in ("anchor", "neighbor", "second-hop"):
        await repository.create_memory(
            MemoryCreate(
                id=memory_id,
                content=f"Memory {memory_id}",
                memory_type=MemoryType.FACT,
                scope_level=ScopeLevel.SCOPE,
                scope_id="alpha",
            )
        )
    await repository.link_support("anchor", "neighbor")
    await repository.link_support("neighbor", "anchor")
    await repository.link_support("neighbor", "second-hop")

    expanded, trace = await bounded_traversal(
        repository,
        ["anchor"],
        allowed_scope_ids={"alpha"},
        max_hops=2,
        max_expanded_nodes=5,
    )

    assert set(expanded) == {"neighbor", "second-hop"}
    assert [(step.from_id, step.to_id, step.depth) for step in trace] == [
        ("anchor", "neighbor", 1),
        ("neighbor", "second-hop", 2),
    ]
    assert trace[-1].path == ["anchor", "neighbor", "second-hop"]


@pytest.mark.asyncio
async def test_embedding_cache_uses_content_and_model_hash(tmp_path: Path) -> None:
    provider = KeywordEmbeddingProvider()
    cache = SQLiteEmbeddingCache(tmp_path / "embeddings.sqlite3")
    embedder = CachedEmbedder(provider, cache)
    try:
        first = await embedder.embed(["Alpha uses Neo4j", "Beta uses MongoDB"])
        second = await embedder.embed(["Alpha uses Neo4j"])
        assert first[0] == second[0]
        assert provider.calls == 1
    finally:
        cache.close()


def test_ranking_requires_normalized_components_and_weights() -> None:
    weights = RankingWeights()
    assert math.isclose(
        final_score(
            semantic=1,
            scope=1,
            temporal=1,
            confidence=1,
            graph=1,
            recency=1,
            weights=weights,
        ),
        1.0,
    )
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    with pytest.raises(ValueError, match="normalized"):
        final_score(
            semantic=2,
            scope=1,
            temporal=1,
            confidence=1,
            graph=1,
            recency=1,
            weights=weights,
        )


def test_token_budget_packing_preserves_rank_order() -> None:
    items = [
        RetrievedMemory(
            memory_id="first",
            content="Alpha uses Neo4j",
            score=1.0,
            scope_id="alpha",
            scope_level=ScopeLevel.SCOPE,
            status=MemoryStatus.ACTIVE,
            confidence=1.0,
        ),
        RetrievedMemory(
            memory_id="second",
            content="A much longer memory that cannot fit",
            score=0.9,
            scope_id="alpha",
            scope_level=ScopeLevel.SCOPE,
            status=MemoryStatus.ACTIVE,
            confidence=1.0,
        ),
    ]
    packed, used = pack_to_token_budget(items, 3)
    assert [item.memory_id for item in packed] == ["first"]
    assert used == 3
