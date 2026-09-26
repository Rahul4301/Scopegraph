import asyncio
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from scopegraph.backends.scopegraph import ScopeGraphMemorySystem
from scopegraph.embeddings.cache import CachedEmbedder, SQLiteEmbeddingCache
from scopegraph.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from scopegraph.graph.in_memory import InMemoryMemoryRepository
from scopegraph.llm.extraction import StaticMemoryExtractor
from scopegraph.llm.transport import ModelTransport
from scopegraph.memory.corrections import CorrectionService
from scopegraph.memory.retriever import ScopeAwareRetriever
from scopegraph.memory.traversal import bounded_traversal
from scopegraph.models.correction import MemoryEditRequest
from scopegraph.models.memory import (
    Memory,
    MemoryCandidate,
    MemoryCreate,
    MemoryStatus,
    MemoryType,
    ScopeLevel,
)
from scopegraph.models.scope import ScopeCreate, ScopeRef, ScopeType
from scopegraph.models.session import SessionInput
from scopegraph.models.source import MessageRole, SourceMessageCreate


async def repository():
    repo = InMemoryMemoryRepository()
    await repo.create_scope(ScopeCreate(id="project", name="Project", scope_type=ScopeType.PROJECT))
    return repo


async def ingest(
    repo, value, *, predicate="uses_database", session="one", temporary=False, date=None
):
    date = date or datetime(2026, 1, 1, tzinfo=UTC)
    candidate = MemoryCandidate(
        content=f"Project {predicate} {value}",
        subject="project",
        predicate=predicate,
        object=value,
        memory_type=MemoryType.FACT,
        confidence=1,
        durability=1,
        proposed_scope_level="session" if temporary else "scope",
        source_message_ids=[session],
    )
    system = ScopeGraphMemorySystem(repo, StaticMemoryExtractor([candidate]))
    return await system.ingest_session(
        SessionInput(
            id=session,
            scope_id="project",
            started_at=date,
            messages=[
                SourceMessageCreate(
                    id=session,
                    session_id=session,
                    role=MessageRole.USER,
                    content=candidate.content,
                    timestamp=date,
                    turn_index=0,
                )
            ],
        ),
        current_scope=ScopeRef(id="project", session_id=session),
    )


@pytest.mark.asyncio
async def test_generic_uses_does_not_erase_database_when_language_arrives():
    repo = await repository()
    await ingest(repo, "Neo4j", predicate="uses", session="db")
    await ingest(repo, "Rust", predicate="uses", session="language")
    memories = await repo.list_memories()
    assert {memory.metadata["object"] for memory in memories} == {"neo4j", "rust"}


@pytest.mark.asyncio
async def test_duplicate_confirmation_preserves_both_sources():
    repo = await repository()
    await ingest(repo, "Neo4j", session="first")
    result = await ingest(repo, "Neo4j", session="second")
    memories = await repo.list_memories()
    assert len(memories) == 1
    assert result.duplicate_count == 1
    assert set(memories[0].source_ids) == {"first", "second"}


@pytest.mark.asyncio
async def test_identical_temporary_values_remain_available_in_each_session():
    repo = await repository()
    await ingest(repo, "SQLite", session="first", temporary=True)
    await ingest(repo, "SQLite", session="second", temporary=True)
    assert len(await repo.list_memories()) == 2
    eligible = await repo.list_retrieval_memories(
        scope_ids={"project"},
        session_id="second",
        historical=False,
        now=datetime.now(UTC),
    )
    assert len(eligible) == 1
    assert eligible[0].metadata["session_id"] == "second"


@pytest.mark.asyncio
async def test_late_arriving_old_fact_cannot_supersede_newer_fact():
    repo = await repository()
    await ingest(repo, "PostgreSQL", session="new", date=datetime(2026, 2, 1, tzinfo=UTC))
    await ingest(repo, "MongoDB", session="old", date=datetime(2026, 1, 1, tzinfo=UTC))
    active = await repo.list_memories()
    assert len(active) == 1 and active[0].metadata["object"] == "postgresql"


@pytest.mark.asyncio
async def test_correction_rolls_back_if_audit_event_cannot_be_saved(monkeypatch):
    repo = await repository()
    result = await ingest(repo, "Neo4j")
    memory_id = result.memory_ids[0]
    original = (await repo.get_memory(memory_id)).model_dump()

    async def fail(*args, **kwargs):
        raise RuntimeError("audit storage unavailable")

    monkeypatch.setattr(repo, "create_correction_event", fail)
    with pytest.raises(RuntimeError, match="audit storage"):
        await CorrectionService(repo).edit(memory_id, MemoryEditRequest(content="Changed"))
    assert (await repo.get_memory(memory_id)).model_dump() == original


@pytest.mark.asyncio
async def test_human_edit_invalidates_old_fact_keys_and_embedding():
    repo = await repository()
    result = await ingest(repo, "Neo4j")
    memory_id = result.memory_ids[0]
    await repo.set_memory_embedding(memory_id, [1, 0], "test")
    corrected = await CorrectionService(repo).edit(
        memory_id, MemoryEditRequest(content="Project uses PostgreSQL")
    )
    assert corrected.memory.embedding is None
    assert "normalized_key" not in corrected.memory.metadata
    assert "object" not in corrected.memory.metadata


@pytest.mark.asyncio
async def test_traversal_cannot_cross_ineligible_bridge():
    repo = await repository()
    for memory_id in ("a", "bridge", "b"):
        await repo.create_memory(
            MemoryCreate(
                id=memory_id,
                content=memory_id,
                memory_type=MemoryType.FACT,
                scope_level=ScopeLevel.SCOPE,
                scope_id="project",
            )
        )
    await repo.link_support("a", "bridge")
    await repo.link_support("bridge", "b")
    expanded, _ = await bounded_traversal(
        repo,
        ["a"],
        allowed_scope_ids={"project"},
        eligible_ids={"a", "b"},
        max_hops=2,
        max_expanded_nodes=10,
    )
    assert expanded == {}


@pytest.mark.asyncio
async def test_traversal_deadline_cancels_slow_database_call():
    class SlowRepository:
        async def get_memory_neighbors(self, *args, **kwargs):
            await asyncio.sleep(5)
            raise AssertionError("Deadline should cancel this call")

    result = await asyncio.wait_for(
        bounded_traversal(
            SlowRepository(),
            ["a"],
            allowed_scope_ids={"project"},
            max_hops=2,
            max_expanded_nodes=10,
            max_time_ms=5,
        ),
        timeout=0.5,
    )
    assert result == ({}, [])


@pytest.mark.asyncio
async def test_embedding_cache_deduplicates_inputs_and_batches_writes(tmp_path):
    class Provider:
        model_name = "test"

        async def embed(self, texts):
            assert texts == ["same", "other"]
            return [[1, 0], [0, 1]]

    cache = SQLiteEmbeddingCache(tmp_path / "vectors.db")
    try:
        embedder = CachedEmbedder(Provider(), cache)
        assert await embedder.embed(["same", "same", "other"]) == [[1, 0], [1, 0], [0, 1]]
        assert await embedder.embed(["same"]) == [[1, 0]]
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_embedding_batches_and_pooled_transport_record_usage():
    sizes = []

    def handler(request):
        import json

        payload = json.loads(request.content)
        sizes.append(len(payload["input"]))
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 3},
                "data": [{"index": i, "embedding": [1, 0]} for i in range(len(payload["input"]))],
            },
        )

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://test.invalid", api_key="test", model="test", batch_size=2
    )
    provider.transport = ModelTransport(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    try:
        assert len(await provider.embed(["a", "b", "c"])) == 3
        assert sizes == [2, 1]
        assert provider.transport.last_usage == {"prompt_tokens": 3}
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_auth_errors_fail_once_without_exposing_secret():
    transport = ModelTransport(
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(401, json={"error": "private response"})
            )
        )
    )
    try:
        with pytest.raises(RuntimeError, match="HTTP 401") as error:
            await transport.post("https://test.invalid", payload={}, api_key="hidden-token")
        assert transport.calls == 1
        assert "hidden-token" not in str(error.value)
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_irrelevant_or_expired_memory_is_not_loaded_or_embedded():
    repo = await repository()
    await repo.create_scope(
        ScopeCreate(id="elsewhere", name="Elsewhere", scope_type=ScopeType.PROJECT)
    )
    for scope_id in ("project", "elsewhere"):
        await repo.create_memory(
            MemoryCreate(
                id=scope_id,
                content="fact",
                memory_type=MemoryType.FACT,
                scope_level=ScopeLevel.SCOPE,
                scope_id=scope_id,
                valid_to=datetime.now(UTC) - timedelta(days=1) if scope_id == "project" else None,
            )
        )

    class NoCalls:
        model_name = "none"

        async def embed(self, texts):
            raise AssertionError("Empty eligible set must not call model API")

    result = await ScopeAwareRetriever(repo, NoCalls()).retrieve(
        "Any facts?",
        current_scope=ScopeRef(id="project"),
        top_k=3,
        token_budget=100,
    )
    assert result.items == []


@pytest.mark.asyncio
async def test_scope_index_follows_move():
    repo = await repository()
    await repo.create_scope(
        ScopeCreate(id="elsewhere", name="Elsewhere", scope_type=ScopeType.PROJECT)
    )
    result = await ingest(repo, "Neo4j")
    await repo.move_memory(result.memory_ids[0], "elsewhere", ScopeLevel.SCOPE)
    assert (
        await repo.list_retrieval_memories(
            scope_ids={"project"}, session_id=None, historical=False, now=datetime.now(UTC)
        )
        == []
    )
    assert (await repo.get_memory(result.memory_ids[0])).status is MemoryStatus.ACTIVE


@pytest.mark.parametrize(
    ("memory_type", "predicate", "flagged", "replaces"),
    [
        (MemoryType.FACT, "uses_database", False, True),
        (MemoryType.FACT, "uses_test_database", False, True),
        (MemoryType.PREFERENCE, "preferred_language", False, True),
        (MemoryType.EVENT, "attended_event", False, False),
        (MemoryType.ENTITY_ATTRIBUTE, "has_pet", False, False),
        (MemoryType.PREFERENCE, "enjoys_activity", False, False),
        (MemoryType.FACT, "uses_art_for", False, False),
        (MemoryType.EVENT, "uses_database", False, False),
        (MemoryType.ENTITY_ATTRIBUTE, "has_pet", True, True),
    ],
)
def test_only_single_valued_attributes_supersede(
    memory_type: MemoryType, predicate: str, flagged: bool, replaces: bool
) -> None:
    from scopegraph.memory.conflict_detector import find_conflicts
    from scopegraph.memory.normalizer import conflict_key, normalize_candidate

    def candidate(value: str) -> MemoryCandidate:
        return normalize_candidate(MemoryCandidate(
            content=f"Subject {predicate} {value}", memory_type=memory_type,
            subject="subject", predicate=predicate, object=value,
            proposed_scope_level="scope", confidence=0.9, durability=0.9,
            source_message_ids=["m"], possible_contradiction=flagged,
        ))

    old = candidate("first")
    existing = Memory(
        id="old", content=old.content, memory_type=memory_type,
        scope_level=ScopeLevel.SCOPE, scope_id="alpha", confidence=0.9,
        metadata={"conflict_key": conflict_key(old), "object": "first"},
    )
    conflicts = find_conflicts(candidate("second"), [existing], scope_id="alpha")
    assert (conflicts == [existing]) is replaces
