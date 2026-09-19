import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from scopegraph.graph.client import Neo4jClient
from scopegraph.models.memory import Memory, MemoryCreate, MemoryUpdate
from scopegraph.models.scope import Scope, ScopeCreate, ScopeType, ScopeUpdate
from scopegraph.models.session import Session, SessionCreate
from scopegraph.models.source import SourceMessage, SourceMessageCreate


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _node(record: dict[str, Any], key: str) -> dict[str, Any]:
    value = record[key]
    return dict(value)


class Neo4jMemoryRepository:
    """Transactional CRUD for Phase 1 graph primitives."""

    def __init__(self, client: Neo4jClient) -> None:
        self.client = client

    async def create_scope(self, request: ScopeCreate) -> Scope:
        if request.scope_type is ScopeType.GLOBAL:
            existing = await self.client.execute_read(
                "MATCH (s:Scope {scope_type: 'global', archived: false}) RETURN s LIMIT 1"
            )
            if existing:
                raise ValueError("An active global root scope already exists")
        payload = request.model_dump(mode="json")
        rows = await self.client.execute_write(
            """
            OPTIONAL MATCH (parent:Scope {id: $parent_scope_id})
            WITH parent
            WHERE $parent_scope_id IS NULL OR parent IS NOT NULL
            CREATE (s:Scope $scope)
            FOREACH (_ IN CASE WHEN parent IS NULL THEN [] ELSE [1] END |
                CREATE (parent)-[:PARENT_OF]->(s))
            RETURN s
            """,
            {"scope": payload, "parent_scope_id": request.parent_scope_id},
        )
        if not rows:
            raise ValueError(f"Parent scope {request.parent_scope_id!r} does not exist")
        return Scope.model_validate(_node(rows[0], "s"))

    async def get_scope(self, scope_id: str) -> Scope | None:
        rows = await self.client.execute_read(
            "MATCH (s:Scope {id: $id}) RETURN s", {"id": scope_id}
        )
        return Scope.model_validate(_node(rows[0], "s")) if rows else None

    async def list_scopes(self, *, include_archived: bool = False) -> list[Scope]:
        rows = await self.client.execute_read(
            """
            MATCH (s:Scope)
            WHERE $include_archived OR s.archived = false
            RETURN s ORDER BY s.created_at, s.name
            """,
            {"include_archived": include_archived},
        )
        return [Scope.model_validate(_node(row, "s")) for row in rows]

    async def update_scope(self, scope_id: str, update: ScopeUpdate) -> Scope | None:
        changes = update.model_dump(mode="json", exclude_unset=True)
        if not changes:
            return await self.get_scope(scope_id)
        if "parent_scope_id" in changes:
            parent_scope_id = changes["parent_scope_id"]
            rows = await self.client.execute_write(
                """
                MATCH (s:Scope {id: $id})
                OPTIONAL MATCH (parent:Scope {id: $parent_scope_id})
                WITH s, parent
                WHERE $parent_scope_id IS NULL OR parent IS NOT NULL
                OPTIONAL MATCH (old_parent:Scope)-[old:PARENT_OF]->(s)
                DELETE old
                SET s += $changes
                FOREACH (_ IN CASE WHEN parent IS NULL THEN [] ELSE [1] END |
                    CREATE (parent)-[:PARENT_OF]->(s))
                RETURN s
                """,
                {
                    "id": scope_id,
                    "parent_scope_id": parent_scope_id,
                    "changes": changes,
                },
            )
            if not rows and await self.get_scope(scope_id) is not None:
                raise ValueError(f"Parent scope {parent_scope_id!r} does not exist")
            return Scope.model_validate(_node(rows[0], "s")) if rows else None
        rows = await self.client.execute_write(
            "MATCH (s:Scope {id: $id}) SET s += $changes RETURN s",
            {"id": scope_id, "changes": changes},
        )
        return Scope.model_validate(_node(rows[0], "s")) if rows else None

    async def create_session(self, request: SessionCreate) -> Session:
        payload = request.model_dump(mode="json")
        payload["metadata_json"] = _json(payload.pop("metadata"))
        rows = await self.client.execute_write(
            """
            MATCH (scope:Scope {id: $scope_id})
            CREATE (s:Session $session)-[:BELONGS_TO]->(scope)
            RETURN s
            """,
            {"scope_id": request.scope_id, "session": payload},
        )
        if not rows:
            raise ValueError(f"Scope {request.scope_id!r} does not exist")
        result = _node(rows[0], "s")
        result["metadata"] = json.loads(result.pop("metadata_json", "{}"))
        return Session.model_validate(result)

    async def get_session(self, session_id: str) -> Session | None:
        rows = await self.client.execute_read(
            "MATCH (s:Session {id: $id}) RETURN s", {"id": session_id}
        )
        if not rows:
            return None
        result = _node(rows[0], "s")
        result["metadata"] = json.loads(result.pop("metadata_json", "{}"))
        return Session.model_validate(result)

    async def create_source_message(self, request: SourceMessageCreate) -> SourceMessage:
        rows = await self.client.execute_write(
            """
            MATCH (session:Session {id: $session_id})
            CREATE (m:SourceMessage $message)-[:PART_OF]->(session)
            RETURN m
            """,
            {"session_id": request.session_id, "message": request.model_dump(mode="json")},
        )
        if not rows:
            raise ValueError(f"Session {request.session_id!r} does not exist")
        return SourceMessage.model_validate(_node(rows[0], "m"))

    async def get_source_message(self, message_id: str) -> SourceMessage | None:
        rows = await self.client.execute_read(
            "MATCH (m:SourceMessage {id: $id}) RETURN m", {"id": message_id}
        )
        return SourceMessage.model_validate(_node(rows[0], "m")) if rows else None

    async def create_memory(self, request: MemoryCreate) -> Memory:
        payload = request.model_dump(mode="json", exclude={"source_ids"})
        payload["metadata_json"] = _json(payload.pop("metadata"))
        rows = await self.client.execute_write(
            """
            MATCH (scope:Scope {id: $scope_id})
            OPTIONAL MATCH (source:SourceMessage)
            WHERE source.id IN $source_ids
            WITH scope, collect(source) AS sources
            WHERE size(sources) = size($source_ids)
            CREATE (m:Memory $memory)-[:BELONGS_TO]->(scope)
            FOREACH (source IN sources | CREATE (m)-[:DERIVED_FROM]->(source))
            RETURN m, [(m)-[:DERIVED_FROM]->(source) | source.id] AS source_ids
            """,
            {
                "scope_id": request.scope_id,
                "memory": payload,
                "source_ids": request.source_ids,
            },
        )
        if not rows:
            raise ValueError("Scope does not exist or a source message was not found")
        return self._memory_from_row(rows[0])

    async def get_memory(self, memory_id: str) -> Memory | None:
        rows = await self.client.execute_read(
            """
            MATCH (m:Memory {id: $id})
            RETURN m, [(m)-[:DERIVED_FROM]->(source) | source.id] AS source_ids
            """,
            {"id": memory_id},
        )
        return self._memory_from_row(rows[0]) if rows else None

    async def list_memories(
        self, *, scope_id: str | None = None, include_inactive: bool = False
    ) -> list[Memory]:
        rows = await self.client.execute_read(
            """
            MATCH (m:Memory)
            WHERE ($scope_id IS NULL OR m.scope_id = $scope_id)
              AND ($include_inactive OR m.status = 'active')
            RETURN m, [(m)-[:DERIVED_FROM]->(source) | source.id] AS source_ids
            ORDER BY m.created_at, m.id
            """,
            {"scope_id": scope_id, "include_inactive": include_inactive},
        )
        return [self._memory_from_row(row) for row in rows]

    async def update_memory(self, memory_id: str, update: MemoryUpdate) -> Memory | None:
        changes = update.model_dump(mode="json", exclude_unset=True)
        if not changes:
            return await self.get_memory(memory_id)
        if "metadata" in changes:
            changes["metadata_json"] = _json(changes.pop("metadata"))
        changes["updated_at"] = datetime.now(UTC).isoformat()
        rows = await self.client.execute_write(
            """
            MATCH (m:Memory {id: $id})
            SET m += $changes, m.revision = m.revision + 1
            RETURN m, [(m)-[:DERIVED_FROM]->(source) | source.id] AS source_ids
            """,
            {"id": memory_id, "changes": changes},
        )
        return self._memory_from_row(rows[0]) if rows else None

    @staticmethod
    def _memory_from_row(row: dict[str, Any]) -> Memory:
        result = _node(row, "m")
        result["metadata"] = json.loads(result.pop("metadata_json", "{}"))
        result["source_ids"] = row.get("source_ids", [])
        return Memory.model_validate(result)
