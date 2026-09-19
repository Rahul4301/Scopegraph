import os
from uuid import uuid4

import pytest

from scopegraph.backends.scopegraph import ScopeGraphMemorySystem
from scopegraph.config import Settings
from scopegraph.graph.client import Neo4jClient
from scopegraph.graph.repository import Neo4jMemoryRepository
from scopegraph.graph.schema import ensure_schema
from scopegraph.llm.extraction import StaticMemoryExtractor
from scopegraph.models.memory import MemoryCandidate, MemoryType
from scopegraph.models.scope import ScopeCreate, ScopeRef, ScopeType
from scopegraph.models.session import SessionInput
from scopegraph.models.source import MessageRole, SourceMessageCreate

pytestmark = pytest.mark.integration


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
