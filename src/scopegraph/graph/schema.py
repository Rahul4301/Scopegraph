from scopegraph.graph.client import Neo4jClient

CONSTRAINTS: tuple[str, ...] = (
    "CREATE CONSTRAINT scope_id_unique IF NOT EXISTS FOR (n:Scope) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT session_id_unique IF NOT EXISTS FOR (n:Session) REQUIRE n.id IS UNIQUE",
    (
        "CREATE CONSTRAINT source_message_id_unique IF NOT EXISTS "
        "FOR (n:SourceMessage) REQUIRE n.id IS UNIQUE"
    ),
    "CREATE CONSTRAINT memory_id_unique IF NOT EXISTS FOR (n:Memory) REQUIRE n.id IS UNIQUE",
    (
        "CREATE CONSTRAINT correction_event_id_unique IF NOT EXISTS "
        "FOR (n:CorrectionEvent) REQUIRE n.id IS UNIQUE"
    ),
    (
        "CREATE CONSTRAINT consolidation_run_id_unique IF NOT EXISTS "
        "FOR (n:ConsolidationRun) REQUIRE n.id IS UNIQUE"
    ),
)

INDEXES: tuple[str, ...] = (
    "CREATE INDEX memory_scope_status_idx IF NOT EXISTS FOR (n:Memory) ON (n.scope_id, n.status)",
    "CREATE INDEX scope_type_idx IF NOT EXISTS FOR (n:Scope) ON (n.scope_type)",
    "CREATE INDEX scope_parent_idx IF NOT EXISTS FOR (n:Scope) ON (n.parent_scope_id)",
    "CREATE INDEX session_scope_idx IF NOT EXISTS FOR (n:Session) ON (n.scope_id)",
    "CREATE INDEX source_session_idx IF NOT EXISTS FOR (n:SourceMessage) ON (n.session_id)",
    "CREATE INDEX memory_scope_idx IF NOT EXISTS FOR (n:Memory) ON (n.scope_id)",
    "CREATE INDEX memory_status_idx IF NOT EXISTS FOR (n:Memory) ON (n.status)",
    "CREATE INDEX memory_type_idx IF NOT EXISTS FOR (n:Memory) ON (n.memory_type)",
    "CREATE INDEX memory_updated_idx IF NOT EXISTS FOR (n:Memory) ON (n.updated_at)",
    (
        "CREATE FULLTEXT INDEX memory_content_fulltext IF NOT EXISTS "
        "FOR (n:Memory) ON EACH [n.content]"
    ),
)


async def ensure_schema(client: Neo4jClient) -> None:
    await client.run_statements((*CONSTRAINTS, *INDEXES))
