import os
from uuid import uuid4

import pytest

from scopegraph.config import Settings
from scopegraph.graph.client import Neo4jClient
from scopegraph.graph.repository import Neo4jMemoryRepository
from scopegraph.graph.schema import ensure_schema
from scopegraph.models.memory import MemoryCreate, MemoryType, ScopeLevel
from scopegraph.models.scope import ScopeCreate, ScopeType
from scopegraph.models.session import SessionCreate
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
        session = await repository.create_session(
            SessionCreate(id=session_id, scope_id=scope_id)
        )
        message = await repository.create_source_message(
            SourceMessageCreate(
                id=message_id,
                session_id=session.id,
                role=MessageRole.USER,
                content="Integration test evidence",
                turn_index=0,
            )
        )
        memory = await repository.create_memory(
            MemoryCreate(
                id=memory_id,
                content="Integration test memory",
                memory_type=MemoryType.FACT,
                scope_level=ScopeLevel.SCOPE,
                scope_id=scope_id,
                source_ids=[message.id],
            )
        )
        assert await repository.get_session(session.id) == session
        assert await repository.get_source_message(message.id) == message
        assert await repository.get_memory(memory.id) == memory
    finally:
        await client.execute_write(
            "MATCH (n) WHERE n.id IN $ids DETACH DELETE n",
            {"ids": [memory_id, message_id, session_id, scope_id]},
        )
        await client.close()
