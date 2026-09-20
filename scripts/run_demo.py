"""Seed a small Neo4j demo and retrieve a scope-aware fact."""

import asyncio
from uuid import uuid4

from scopegraph.config import get_settings
from scopegraph.graph.client import Neo4jClient
from scopegraph.graph.repository import Neo4jMemoryRepository
from scopegraph.graph.schema import ensure_schema
from scopegraph.llm.extraction import StaticMemoryExtractor
from scopegraph.memory.retriever import ScopeAwareRetriever
from scopegraph.models.memory import MemoryCandidate, MemoryType
from scopegraph.models.scope import ScopeCreate, ScopeRef, ScopeType
from scopegraph.models.session import SessionInput
from scopegraph.models.source import MessageRole, SourceMessageCreate


class DemoEmbeddingProvider:
    model_name = "demo-v1"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


async def main() -> None:
    client = Neo4jClient(get_settings())
    try:
        if not await client.health():
            raise SystemExit("Neo4j is unavailable; start it with `make neo4j-up`")
        await ensure_schema(client)
        repository = Neo4jMemoryRepository(client)
        global_scope = await repository.get_global_scope()
        if global_scope is None:
            global_scope = await repository.create_scope(
                ScopeCreate(id="demo-global", name="Global", scope_type=ScopeType.GLOBAL)
            )
        scope = await repository.get_scope("demo-project")
        if scope is None:
            scope = await repository.create_scope(
                ScopeCreate(
                    id="demo-project", name="Demo Project", scope_type=ScopeType.PROJECT,
                    parent_scope_id=global_scope.id,
                )
            )
        session_id = f"demo-session-{uuid4()}"
        message_id = f"demo-message-{uuid4()}"
        candidate = MemoryCandidate(
            content="Demo Project uses Neo4j",
            memory_type=MemoryType.FACT,
            subject="demo project",
            predicate="uses_database",
            object="Neo4j",
            proposed_scope_level="scope",
            confidence=1.0,
            durability=1.0,
            source_message_ids=[message_id],
        )
        from scopegraph.backends.scopegraph import ScopeGraphMemorySystem

        system = ScopeGraphMemorySystem(
            repository,
            StaticMemoryExtractor([candidate]),
            retriever=ScopeAwareRetriever(repository, DemoEmbeddingProvider()),
        )
        await system.ingest_session(
            SessionInput(
                id=session_id,
                scope_id=scope.id,
                messages=[SourceMessageCreate(
                    id=message_id, session_id=session_id, role=MessageRole.USER,
                    content="Demo Project uses Neo4j", turn_index=0,
                )],
            ),
            current_scope=ScopeRef(
                id=scope.id, name=scope.name, scope_type=scope.scope_type, session_id=session_id,
            ),
        )
        result = await system.retrieve(
            "Which database does Demo Project use?",
            current_scope=ScopeRef(id=scope.id, session_id=session_id),
            top_k=3,
            token_budget=100,
        )
        print(result.model_dump_json(indent=2))
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
