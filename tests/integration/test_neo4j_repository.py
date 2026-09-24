import os
from uuid import uuid4

import pytest

from scopegraph.backends.scopegraph import ScopeGraphMemorySystem
from scopegraph.config import Settings
from scopegraph.graph.client import Neo4jClient
from scopegraph.graph.repository import Neo4jMemoryRepository
from scopegraph.graph.schema import ensure_schema
from scopegraph.llm.extraction import StaticMemoryExtractor
from scopegraph.memory.corrections import CorrectionService
from scopegraph.memory.retriever import ScopeAwareRetriever
from scopegraph.models.correction import (
    CorrectionRelation,
    MemoryEditRequest,
    MemoryRestoreRequest,
    RelationCorrectionRequest,
)
from scopegraph.models.memory import MemoryCandidate, MemoryCreate, MemoryType, ScopeLevel
from scopegraph.models.relationship import RelationKind
from scopegraph.models.scope import ScopeCreate, ScopeRef, ScopeType
from scopegraph.models.session import SessionInput
from scopegraph.models.source import MessageRole, SourceMessageCreate

pytestmark = pytest.mark.integration


class IntegrationEmbeddingProvider:
    model_name = "integration-test-v1"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


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
    related_memory_id = f"integration-related-memory-{run_id}"
    correction_ids: list[str] = []
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
        assert memory_id in [item.memory_id for item in result.items]
        assert result.trace[0].relation == "SEMANTIC_ANCHOR"
        persisted = await repository.get_memory(memory_id)
        assert persisted is not None
        assert persisted.embedding == [1.0, 0.0]
        assert persisted.embedding_model == "integration-test-v1"

        corrections = CorrectionService(repository)
        edited = await corrections.edit(
            memory_id,
            MemoryEditRequest(
                content="Integration scope uses Neo4j Community",
                actor="integration-test",
                reason="verify revision history",
            ),
        )
        correction_ids.append(edited.event.id)
        assert edited.memory.revision == 2
        pruned = await corrections.prune(memory_id, actor="integration-test")
        correction_ids.append(pruned.event.id)
        assert pruned.memory.status.value == "tombstoned"
        restored = await corrections.restore(
            memory_id,
            MemoryRestoreRequest(
                undo_of=pruned.event.id,
                actor="integration-test",
            ),
        )
        correction_ids.append(restored.event.id)
        assert restored.memory.status.value == "active"

        await repository.create_memory(
            MemoryCreate(
                id=related_memory_id,
                content="Neo4j Community is a graph database",
                memory_type=MemoryType.FACT,
                scope_level=ScopeLevel.SCOPE,
                scope_id=scope_id,
            )
        )
        relation_request = RelationCorrectionRequest(
            target_memory_id=related_memory_id,
            relation=CorrectionRelation.RELATES_TO,
            kind=RelationKind.USES,
            actor="integration-test",
        )
        added_relation = await corrections.add_relation(memory_id, relation_request)
        correction_ids.append(added_relation.event.id)

        graph = await repository.get_subgraph(
            scope_id=scope_id, include_inactive=True, include_sources=True
        )
        assert {memory_id, related_memory_id, message_id, scope_id}.issubset(
            {node.id for node in graph.nodes}
        )
        assert {"BELONGS_TO", "DERIVED_FROM", "RELATES_TO"}.issubset(
            {edge.relation for edge in graph.edges}
        )
        focused = await repository.get_subgraph(memory_id=memory_id)
        assert {memory_id, related_memory_id}.issubset(
            {node.id for node in focused.nodes}
        )

        removed_relation = await corrections.remove_relation(memory_id, relation_request)
        correction_ids.append(removed_relation.event.id)
        assert [event.id for event in await corrections.history(memory_id)] == correction_ids
    finally:
        await client.execute_write(
            "MATCH (event:CorrectionEvent) WHERE event.id IN $ids DETACH DELETE event",
            {"ids": correction_ids},
        )
        await client.execute_write(
            "MATCH (m:Memory) WHERE m.metadata_json CONTAINS $session_id DETACH DELETE m",
            {"session_id": session_id},
        )
        await client.execute_write(
            "MATCH (n) WHERE n.id IN $ids DETACH DELETE n",
            {
                "ids": [
                    memory_id,
                    related_memory_id,
                    message_id,
                    session_id,
                    scope_id,
                ]
            },
        )
        await client.close()
