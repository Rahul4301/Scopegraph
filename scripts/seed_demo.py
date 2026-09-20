"""Seed a rich, repeatable Neo4j workspace for the Memory Explorer demo."""

import asyncio
from datetime import UTC, datetime, timedelta

from scopegraph.config import get_settings
from scopegraph.graph.client import Neo4jClient
from scopegraph.graph.repository import Neo4jMemoryRepository
from scopegraph.graph.schema import ensure_schema
from scopegraph.models.correction import CorrectionRelation
from scopegraph.models.memory import MemoryCreate, MemoryStatus, MemoryType, ScopeLevel
from scopegraph.models.relationship import RelationKind
from scopegraph.models.scope import ScopeCreate, ScopeType
from scopegraph.models.session import SessionCreate
from scopegraph.models.source import MessageRole, SourceMessageCreate


def _embedding(topic: str) -> list[float]:
    """Deterministic demo vectors; live runs replace these with the configured provider."""
    buckets = [0.0] * 8
    for index, char in enumerate(topic.lower()):
        buckets[(ord(char) + index) % len(buckets)] += 1.0
    magnitude = sum(value * value for value in buckets) ** 0.5 or 1.0
    return [round(value / magnitude, 6) for value in buckets]


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

        scope_specs = [
            ("atlas-workspace", "Atlas workspace", ScopeType.WORKSPACE, global_scope.id),
            ("atlas-api", "Atlas API", ScopeType.REPOSITORY, "atlas-workspace"),
            ("atlas-web", "Atlas Web", ScopeType.REPOSITORY, "atlas-workspace"),
            ("atlas-ops", "Atlas Operations", ScopeType.TASK, "atlas-workspace"),
            ("northstar-project", "Northstar (unrelated)", ScopeType.PROJECT, global_scope.id),
        ]
        for scope_id, name, scope_type, parent_id in scope_specs:
            if await repository.get_scope(scope_id) is None:
                await repository.create_scope(
                    ScopeCreate(
                        id=scope_id, name=name, scope_type=scope_type, parent_scope_id=parent_id
                    )
                )

        now = datetime.now(UTC)
        old = now - timedelta(days=45)
        session_specs = {
            "demo-global": ("demo-session-global", "Global memory policy for the Atlas workspace."),
            "atlas-api": (
                "demo-session-atlas-api",
                "The API team is deciding how to persist context.",
            ),
            "atlas-web": ("demo-session-atlas-web", "The web team is wiring the memory console."),
            "atlas-ops": (
                "demo-session-atlas-ops",
                "Operations is validating deployment guardrails.",
            ),
            "northstar-project": ("demo-session-northstar", "Northstar uses a separate stack."),
        }
        for scope_id, (session_id, opening) in session_specs.items():
            if await repository.get_session(session_id) is None:
                await repository.create_session(SessionCreate(id=session_id, scope_id=scope_id))
                for turn, content in enumerate(
                    (opening, "Capture the durable decision and keep the old answer inspectable.")
                ):
                    await repository.create_source_message(
                        SourceMessageCreate(
                            id=f"{session_id}-m{turn + 1}",
                            session_id=session_id,
                            role=MessageRole.USER,
                            content=content,
                            turn_index=turn,
                        )
                    )

        records = [
            (
                "demo-global-model",
                "The team prefers small, inspectable memory changes over opaque rewrites.",
                MemoryType.PREFERENCE,
                ScopeLevel.GLOBAL,
                "demo-global",
                0.96,
                "demo-session-global-m1",
                now,
                None,
            ),
            (
                "demo-global-privacy",
                "Personal data must not be promoted into global memory without an explicit signal.",
                MemoryType.INSTRUCTION,
                ScopeLevel.GLOBAL,
                "demo-global",
                0.99,
                "demo-session-global-m1",
                now,
                None,
            ),
            (
                "demo-atlas-architecture",
                "Atlas uses ScopeGraph to separate workspace, repository, session, "
                "and global memory.",
                MemoryType.DECISION,
                ScopeLevel.SCOPE,
                "atlas-workspace",
                0.98,
                "demo-session-atlas-api-m1",
                now,
                None,
            ),
            (
                "demo-atlas-api-old-db",
                "Atlas API used PostgreSQL for memory persistence.",
                MemoryType.FACT,
                ScopeLevel.SCOPE,
                "atlas-api",
                0.72,
                "demo-session-atlas-api-m1",
                old,
                old + timedelta(days=30),
            ),
            (
                "demo-atlas-api-db",
                "Atlas API uses Neo4j for memory persistence.",
                MemoryType.DECISION,
                ScopeLevel.SCOPE,
                "atlas-api",
                0.98,
                "demo-session-atlas-api-m2",
                now,
                None,
            ),
            (
                "demo-atlas-api-embedding",
                "Atlas API uses a cached embedding provider and bounded cosine ranking.",
                MemoryType.FACT,
                ScopeLevel.SCOPE,
                "atlas-api",
                0.94,
                "demo-session-atlas-api-m2",
                now,
                None,
            ),
            (
                "demo-atlas-api-boundary",
                "API retrieval may access Atlas API, Atlas workspace, and Global scopes, "
                "but not Northstar.",
                MemoryType.INSTRUCTION,
                ScopeLevel.SCOPE,
                "atlas-api",
                0.97,
                "demo-session-atlas-api-m2",
                now,
                None,
            ),
            (
                "demo-atlas-web-stack",
                "Atlas Web uses React and Cytoscape for the inspectable graph console.",
                MemoryType.FACT,
                ScopeLevel.SCOPE,
                "atlas-web",
                0.95,
                "demo-session-atlas-web-m1",
                now,
                None,
            ),
            (
                "demo-atlas-web-query",
                "The retrieval debugger exposes score components, token budget, latency, "
                "and traversal path.",
                MemoryType.FACT,
                ScopeLevel.SCOPE,
                "atlas-web",
                0.97,
                "demo-session-atlas-web-m2",
                now,
                None,
            ),
            (
                "demo-atlas-ops-health",
                "Atlas Operations keeps Neo4j local and checks health before serving retrieval.",
                MemoryType.INSTRUCTION,
                ScopeLevel.SCOPE,
                "atlas-ops",
                0.93,
                "demo-session-atlas-ops-m1",
                now,
                None,
            ),
            (
                "demo-atlas-session-query",
                "The current API session is testing whether graph edges recover the database "
                "decision.",
                MemoryType.TASK_STATE,
                ScopeLevel.SESSION,
                "atlas-api",
                0.89,
                "demo-session-atlas-api-m2",
                now,
                None,
            ),
            (
                "demo-northstar-db",
                "Northstar uses SQLite for its local prototype.",
                MemoryType.FACT,
                ScopeLevel.SCOPE,
                "northstar-project",
                0.99,
                "demo-session-northstar-m1",
                now,
                None,
            ),
            (
                "demo-northstar-cloud",
                "Northstar deploys on a separate cloud account and must not leak into Atlas "
                "answers.",
                MemoryType.INSTRUCTION,
                ScopeLevel.SCOPE,
                "northstar-project",
                0.96,
                "demo-session-northstar-m2",
                now,
                None,
            ),
        ]
        for (
            memory_id,
            content,
            memory_type,
            level,
            scope_id,
            confidence,
            message_id,
            valid_from,
            valid_to,
        ) in records:
            if await repository.get_memory(memory_id) is None:
                await repository.create_memory(
                    MemoryCreate(
                        id=memory_id,
                        content=content,
                        memory_type=memory_type,
                        scope_level=level,
                        scope_id=scope_id,
                        confidence=confidence,
                        status=MemoryStatus.ACTIVE if valid_to is None else MemoryStatus.SUPERSEDED,
                        valid_from=valid_from,
                        valid_to=valid_to,
                        last_confirmed_at=valid_from,
                        embedding=_embedding(content),
                        embedding_model="demo-hash-v1",
                        source_ids=[message_id],
                        metadata={"demo": True, "fixture": "atlas-workspace"},
                    )
                )

        relations = [
            ("demo-atlas-api-db", "demo-atlas-api-old-db", CorrectionRelation.CONTRADICTS, None),
            ("demo-atlas-api-db", "demo-atlas-architecture", CorrectionRelation.SUPPORTS, None),
            (
                "demo-atlas-api-embedding",
                "demo-atlas-api-db",
                CorrectionRelation.RELATES_TO,
                RelationKind.DEPENDS_ON,
            ),
            (
                "demo-atlas-web-query",
                "demo-atlas-architecture",
                CorrectionRelation.RELATES_TO,
                RelationKind.USES,
            ),
            (
                "demo-atlas-ops-health",
                "demo-atlas-api-db",
                CorrectionRelation.RELATES_TO,
                RelationKind.DEPENDS_ON,
            ),
            ("demo-atlas-session-query", "demo-atlas-api-db", CorrectionRelation.SUPPORTS, None),
            ("demo-atlas-web-stack", "demo-atlas-web-query", CorrectionRelation.SUPPORTS, None),
            ("demo-global-privacy", "demo-atlas-api-boundary", CorrectionRelation.SUPPORTS, None),
        ]
        for source_id, target_id, relation, kind in relations:
            await repository.add_memory_relation(source_id, target_id, relation, kind)

        print(
            "Seeded Atlas demo workspace: 6 scopes, 4 sessions, 13 memories, and 8 graph relations."
        )
        print("Try: Which database does Atlas API use? (scope: Atlas API)")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
