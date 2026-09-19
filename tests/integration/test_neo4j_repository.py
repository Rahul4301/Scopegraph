import os
from uuid import uuid4

import pytest

from scopegraph.backends.flat_graph import FlatGraphMemory
from scopegraph.backends.scopegraph import ScopeGraphMemorySystem
from scopegraph.backends.two_level_graph import TwoLevelGraphMemory
from scopegraph.backends.vector_memory import VectorMemory
from scopegraph.config import Settings
from scopegraph.graph.client import Neo4jClient
from scopegraph.graph.repository import Neo4jMemoryRepository
from scopegraph.graph.schema import ensure_schema
from scopegraph.llm.extraction import StaticMemoryExtractor
from scopegraph.memory.retriever import ScopeAwareRetriever
from scopegraph.models.memory import MemoryCandidate, MemoryType
from scopegraph.models.scope import ScopeCreate, ScopeRef, ScopeType
from scopegraph.models.session import SessionInput
from scopegraph.models.source import MessageRole, SourceMessageCreate

pytestmark = pytest.mark.integration


class IntegrationEmbeddingProvider:
    model_name = "integration-test-v1"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


@pytest.mark.parametrize(
    ("backend_type", "backend_name"),
    [
        (VectorMemory, "vector_memory"),
        (FlatGraphMemory, "flat_graph"),
        (TwoLevelGraphMemory, "two_level_graph"),
    ],
)
@pytest.mark.asyncio
async def test_baseline_neo4j_round_trip(backend_type: type, backend_name: str) -> None:
    if os.getenv("SCOPEGRAPH_RUN_INTEGRATION") != "1":
        pytest.skip("Set SCOPEGRAPH_RUN_INTEGRATION=1 with a disposable Neo4j database")
    client = Neo4jClient(Settings())
    run_id = str(uuid4())
    global_id = f"integration-global-{run_id}"
    scope_id = f"integration-baseline-scope-{run_id}"
    session_id = f"integration-baseline-session-{run_id}"
    message_id = f"integration-baseline-message-{run_id}"
    memory_ids: list[str] = []
    try:
        assert await client.health()
        await ensure_schema(client)
        repository = Neo4jMemoryRepository(client)
        await repository.create_scope(
            ScopeCreate(id=global_id, name="Global", scope_type=ScopeType.GLOBAL)
        )
        await repository.create_scope(
            ScopeCreate(
                id=scope_id,
                name="Baseline integration",
                scope_type=ScopeType.PROJECT,
                parent_scope_id=global_id,
            )
        )
        extractor = StaticMemoryExtractor(
            [
                MemoryCandidate(
                    content="Baseline integration uses Neo4j",
                    memory_type=MemoryType.FACT,
                    subject="baseline integration",
                    predicate="uses_database",
                    object="Neo4j",
                    proposed_scope_level="scope",
                    confidence=1.0,
                    durability=1.0,
                    source_message_ids=[message_id],
                )
            ]
        )
        system = backend_type(repository, extractor, IntegrationEmbeddingProvider())
        ingest = await system.ingest_session(
            SessionInput(
                id=session_id,
                scope_id=scope_id,
                messages=[
                    SourceMessageCreate(
                        id=message_id,
                        session_id=session_id,
                        role=MessageRole.USER,
                        content="Baseline integration uses Neo4j",
                        turn_index=0,
                    )
                ],
            ),
            current_scope=ScopeRef(
                id=scope_id,
                name="Baseline integration",
                scope_type=ScopeType.PROJECT,
                session_id=session_id,
            ),
        )
        memory_ids = ingest.memory_ids
        result = await system.retrieve(
            "Which database does baseline integration use?",
            current_scope=ScopeRef(id=scope_id, session_id=session_id),
            top_k=3,
            token_budget=100,
        )
        assert result.backend_name == backend_name
        assert [item.memory_id for item in result.items] == memory_ids
    finally:
        await client.execute_write(
            "MATCH (n) WHERE n.id IN $ids DETACH DELETE n",
            {"ids": [*memory_ids, message_id, session_id, scope_id, global_id]},
        )
        await client.close()


@pytest.mark.asyncio
async def test_neo4j_scope_round_trip() -> None:
    if os.getenv("SCOPEGRAPH_RUN_INTEGRATION") != "1":
        pytest.skip("Set SCOPEGRAPH_RUN_INTEGRATION=1 with a disposable Neo4j database")
    client = Neo4jClient(Settings())
    run_id = str(uuid4())
    scope_id = f"integration-scope-{run_id}"
    session_id = f"integration-session-{run_id}"
    message_id = f"integration-message-{run_id}"
    memory_id = f"integration-memory-{run_id}"
    try:
        assert await client.health()
        await ensure_schema(client)
        repository = Neo4jMemoryRepository(client)
        created = await repository.create_scope(
            ScopeCreate(id=scope_id, name="Integration scope", scope_type=ScopeType.CUSTOM)
        )
        assert await repository.get_scope(created.id) == created
        candidate = MemoryCandidate(
            content="Integration scope uses Neo4j",
            memory_type=MemoryType.FACT,
            subject="integration scope",
            predicate="uses_database",
            object="Neo4j",
            proposed_scope_level="scope",
            confidence=1.0,
            durability=1.0,
            source_message_ids=[message_id],
        )
        system = ScopeGraphMemorySystem(
            repository, StaticMemoryExtractor([candidate])
        )
        ingest = await system.ingest_session(
            SessionInput(
                id=session_id,
                scope_id=scope_id,
                messages=[
                    SourceMessageCreate(
                        id=message_id,
                        session_id=session_id,
                        role=MessageRole.USER,
                        content="Integration scope uses Neo4j",
                        turn_index=0,
                    )
                ],
            ),
            current_scope=ScopeRef(id=scope_id, scope_type=ScopeType.CUSTOM),
        )
        memory_id = ingest.memory_ids[0]
        assert await repository.get_session(session_id) is not None
        assert await repository.get_source_message(message_id) is not None
        memory = await repository.get_memory(memory_id)
        assert memory is not None
        assert memory.source_ids == [message_id]

        retriever = ScopeAwareRetriever(repository, IntegrationEmbeddingProvider())
        result = await retriever.retrieve(
            "Which database does this scope use?",
            current_scope=ScopeRef(id=scope_id, scope_type=ScopeType.CUSTOM),
            top_k=3,
            token_budget=100,
        )
        assert [item.memory_id for item in result.items] == [memory_id]
        assert result.trace[0].relation == "SEMANTIC_ANCHOR"
        persisted = await repository.get_memory(memory_id)
        assert persisted is not None
        assert persisted.embedding == [1.0, 0.0]
        assert persisted.embedding_model == "integration-test-v1"
    finally:
        await client.execute_write(
            "MATCH (m:Memory) WHERE m.metadata_json CONTAINS $session_id DETACH DELETE m",
            {"session_id": session_id},
        )
        await client.execute_write(
            "MATCH (n) WHERE n.id IN $ids DETACH DELETE n",
            {"ids": [memory_id, message_id, session_id, scope_id]},
        )
        await client.close()
