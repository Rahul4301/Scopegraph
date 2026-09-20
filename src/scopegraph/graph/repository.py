import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from scopegraph.graph.client import Neo4jClient
from scopegraph.memory.traversal import MemoryNeighbor
from scopegraph.models.correction import (
    CorrectionEvent,
    CorrectionRelation,
    SupportDependency,
)
from scopegraph.models.graph import GraphEdge, GraphNode, GraphSubgraph
from scopegraph.models.memory import Memory, MemoryCreate, MemoryUpdate, ScopeLevel
from scopegraph.models.relationship import RelationKind
from scopegraph.models.retrieval import MemoryStats
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

    async def get_global_scope(self) -> Scope | None:
        rows = await self.client.execute_read(
            "MATCH (s:Scope {scope_type: 'global', archived: false}) RETURN s LIMIT 1"
        )
        return Scope.model_validate(_node(rows[0], "s")) if rows else None

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

    async def end_session(self, session_id: str, ended_at: datetime) -> Session | None:
        rows = await self.client.execute_write(
            "MATCH (s:Session {id: $id}) SET s.ended_at = $ended_at RETURN s",
            {"id": session_id, "ended_at": ended_at.isoformat()},
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

    async def list_source_messages(self, session_id: str) -> list[SourceMessage]:
        rows = await self.client.execute_read(
            """
            MATCH (m:SourceMessage)-[:PART_OF]->(:Session {id: $session_id})
            RETURN m ORDER BY m.turn_index, m.timestamp
            """,
            {"session_id": session_id},
        )
        return [SourceMessage.model_validate(_node(row, "m")) for row in rows]

    async def get_source_messages_by_ids(
        self, message_ids: list[str]
    ) -> list[SourceMessage]:
        if not message_ids:
            return []
        rows = await self.client.execute_read(
            """
            MATCH (message:SourceMessage)
            WHERE message.id IN $message_ids
            RETURN message AS m
            ORDER BY message.timestamp, message.turn_index, message.id
            """,
            {"message_ids": message_ids},
        )
        return [SourceMessage.model_validate(_node(row, "m")) for row in rows]

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

    async def set_memory_embedding(
        self, memory_id: str, embedding: list[float], model_name: str
    ) -> None:
        rows = await self.client.execute_write(
            """
            MATCH (m:Memory {id: $id})
            SET m.embedding = $embedding, m.embedding_model = $model_name
            RETURN m.id AS id
            """,
            {"id": memory_id, "embedding": embedding, "model_name": model_name},
        )
        if not rows:
            raise ValueError(f"Memory {memory_id!r} does not exist")

    async def get_memory_neighbors(self, memory_ids: list[str]) -> list[MemoryNeighbor]:
        if not memory_ids:
            return []
        rows = await self.client.execute_read(
            """
            MATCH (source:Memory)-[r:SUPERSEDES|CONTRADICTS|SAME_AS|SUPPORTS|RELATES_TO]-(m:Memory)
            WHERE source.id IN $memory_ids
            RETURN source.id AS source_id, type(r) AS relation, m,
                   [(m)-[:DERIVED_FROM]->(message) | message.id] AS source_ids
            ORDER BY source.id, m.id
            """,
            {"memory_ids": memory_ids},
        )
        return [
            MemoryNeighbor(
                source_id=row["source_id"],
                relation=row["relation"],
                memory=self._memory_from_row(row),
            )
            for row in rows
        ]

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

    async def move_memory(
        self, memory_id: str, scope_id: str, scope_level: ScopeLevel
    ) -> Memory | None:
        rows = await self.client.execute_write(
            """
            MATCH (m:Memory {id: $memory_id}), (scope:Scope {id: $scope_id})
            OPTIONAL MATCH (m)-[old:BELONGS_TO]->(:Scope)
            DELETE old
            SET m.scope_id = $scope_id,
                m.scope_level = $scope_level,
                m.updated_at = $updated_at,
                m.revision = m.revision + 1
            MERGE (m)-[:BELONGS_TO]->(scope)
            RETURN m, [(m)-[:DERIVED_FROM]->(source) | source.id] AS source_ids
            """,
            {
                "memory_id": memory_id,
                "scope_id": scope_id,
                "scope_level": scope_level.value,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        if not rows:
            if await self.get_memory(memory_id) is None:
                return None
            raise ValueError(f"Scope {scope_id!r} does not exist")
        return self._memory_from_row(rows[0])

    async def merge_memories(
        self, source_memory_id: str, target_memory_id: str
    ) -> tuple[Memory, Memory]:
        rows = await self.client.execute_write(
            """
            MATCH (source:Memory {id: $source_id}), (target:Memory {id: $target_id})
            OPTIONAL MATCH (source)-[:DERIVED_FROM]->(message:SourceMessage)
            WITH source, target, collect(message) AS messages
            FOREACH (message IN messages | MERGE (target)-[:DERIVED_FROM]->(message))
            SET source.status = 'tombstoned',
                source.updated_at = $updated_at,
                source.revision = source.revision + 1,
                target.updated_at = $updated_at,
                target.revision = target.revision + 1
            MERGE (source)-[:SAME_AS]->(target)
            RETURN source AS source_memory,
                   [(source)-[:DERIVED_FROM]->(message) | message.id] AS source_ids,
                   target AS target_memory,
                   [(target)-[:DERIVED_FROM]->(message) | message.id] AS target_source_ids
            """,
            {
                "source_id": source_memory_id,
                "target_id": target_memory_id,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        if not rows:
            raise ValueError("Both memories must exist to merge")
        row = rows[0]
        source_row = {"m": row["source_memory"], "source_ids": row["source_ids"]}
        target_row = {
            "m": row["target_memory"],
            "source_ids": row["target_source_ids"],
        }
        return self._memory_from_row(source_row), self._memory_from_row(target_row)

    async def add_memory_relation(
        self,
        source_memory_id: str,
        target_memory_id: str,
        relation: CorrectionRelation,
        kind: RelationKind | None = None,
    ) -> Memory:
        if source_memory_id == target_memory_id:
            raise ValueError("A memory cannot relate to itself")
        patterns = {
            CorrectionRelation.SUPPORTS: "MERGE (source)-[:SUPPORTS]->(target)",
            CorrectionRelation.SAME_AS: "MERGE (source)-[:SAME_AS]->(target)",
            CorrectionRelation.CONTRADICTS: "MERGE (source)-[:CONTRADICTS]->(target)",
            CorrectionRelation.RELATES_TO: (
                "MERGE (source)-[:RELATES_TO {kind: $kind}]->(target)"
            ),
        }
        if relation is CorrectionRelation.RELATES_TO and kind is None:
            raise ValueError("RELATES_TO requires an allowlisted kind")
        rows = await self.client.execute_write(
            f"""
            MATCH (source:Memory {{id: $source_id}}), (target:Memory {{id: $target_id}})
            {patterns[relation]}
            SET source.updated_at = $updated_at, source.revision = source.revision + 1
            RETURN source AS m,
                   [(source)-[:DERIVED_FROM]->(message) | message.id] AS source_ids
            """,
            {
                "source_id": source_memory_id,
                "target_id": target_memory_id,
                "kind": kind.value if kind else None,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        if not rows:
            raise ValueError("Both memories must exist to add a relation")
        return self._memory_from_row(rows[0])

    async def remove_memory_relation(
        self,
        source_memory_id: str,
        target_memory_id: str,
        relation: CorrectionRelation,
        kind: RelationKind | None = None,
    ) -> Memory:
        patterns = {
            CorrectionRelation.SUPPORTS: "[relation:SUPPORTS]",
            CorrectionRelation.SAME_AS: "[relation:SAME_AS]",
            CorrectionRelation.CONTRADICTS: "[relation:CONTRADICTS]",
            CorrectionRelation.RELATES_TO: "[relation:RELATES_TO {kind: $kind}]",
        }
        if relation is CorrectionRelation.RELATES_TO and kind is None:
            raise ValueError("RELATES_TO requires an allowlisted kind")
        rows = await self.client.execute_write(
            f"""
            MATCH (source:Memory {{id: $source_id}})-{patterns[relation]}->
                  (target:Memory {{id: $target_id}})
            DELETE relation
            SET source.updated_at = $updated_at, source.revision = source.revision + 1
            RETURN source AS m,
                   [(source)-[:DERIVED_FROM]->(message) | message.id] AS source_ids
            """,
            {
                "source_id": source_memory_id,
                "target_id": target_memory_id,
                "kind": kind.value if kind else None,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        if not rows:
            raise ValueError("The requested memory relation does not exist")
        return self._memory_from_row(rows[0])

    async def get_support_dependents(self, memory_id: str) -> list[SupportDependency]:
        rows = await self.client.execute_read(
            """
            MATCH (:Memory {id: $memory_id})-[:SUPPORTS]->(dependent:Memory)
            OPTIONAL MATCH (other:Memory)-[:SUPPORTS]->(dependent)
            WHERE other.id <> $memory_id AND other.status = 'active'
            RETURN dependent AS m,
                   [(dependent)-[:DERIVED_FROM]->(source) | source.id] AS source_ids,
                   [id IN collect(DISTINCT other.id) WHERE id IS NOT NULL]
                       AS other_active_support_ids
            ORDER BY dependent.id
            """,
            {"memory_id": memory_id},
        )
        return [
            SupportDependency(
                memory=self._memory_from_row(row),
                other_active_support_ids=row["other_active_support_ids"],
            )
            for row in rows
        ]

    async def get_subgraph(
        self,
        *,
        scope_id: str | None = None,
        memory_id: str | None = None,
        include_inactive: bool = True,
        include_sources: bool = False,
        limit: int = 200,
    ) -> GraphSubgraph:
        scopes = await self.list_scopes(include_archived=True)
        rows = await self.client.execute_read(
            """
            MATCH (m:Memory)
            WHERE ($scope_id IS NULL OR m.scope_id = $scope_id OR $memory_id IS NOT NULL)
              AND ($include_inactive OR m.status = 'active')
              AND (
                $memory_id IS NULL OR m.id = $memory_id OR EXISTS {
                  MATCH (m)-[:SUPERSEDES|CONTRADICTS|SAME_AS|SUPPORTS|RELATES_TO]-
                        (:Memory {id: $memory_id})
                }
              )
            RETURN m, [(m)-[:DERIVED_FROM]->(source) | source.id] AS source_ids
            ORDER BY m.created_at, m.id
            LIMIT $limit
            """,
            {
                "scope_id": scope_id,
                "memory_id": memory_id,
                "include_inactive": include_inactive,
                "limit": limit,
            },
        )
        memories = [self._memory_from_row(row) for row in rows]
        memory_ids = [memory.id for memory in memories]
        nodes: dict[str, GraphNode] = {
            scope.id: GraphNode(
                id=scope.id,
                node_type="scope",
                label=scope.name,
                data=scope.model_dump(mode="json"),
            )
            for scope in scopes
        }
        edges: dict[str, GraphEdge] = {}
        for scope in scopes:
            if scope.parent_scope_id:
                edge = GraphEdge(
                    id=f"scope:{scope.parent_scope_id}:PARENT_OF:{scope.id}",
                    source=scope.parent_scope_id,
                    target=scope.id,
                    relation="PARENT_OF",
                )
                edges[edge.id] = edge
        for memory in memories:
            nodes[memory.id] = GraphNode(
                id=memory.id,
                node_type="memory",
                label=memory.content[:64],
                data=memory.model_dump(mode="json"),
            )
            edge = GraphEdge(
                id=f"memory:{memory.id}:BELONGS_TO:{memory.scope_id}",
                source=memory.id,
                target=memory.scope_id,
                relation="BELONGS_TO",
            )
            edges[edge.id] = edge
        if memory_ids:
            relationship_rows = await self.client.execute_read(
                """
                MATCH (source:Memory)-
                      [relation:SUPERSEDES|CONTRADICTS|SAME_AS|SUPPORTS|RELATES_TO]->
                      (target:Memory)
                WHERE source.id IN $memory_ids AND target.id IN $memory_ids
                RETURN source.id AS source, target.id AS target,
                       type(relation) AS relation, relation.kind AS kind
                ORDER BY source.id, relation, target.id
                """,
                {"memory_ids": memory_ids},
            )
            for row in relationship_rows:
                edge = GraphEdge(
                    id=(
                        f"memory:{row['source']}:{row['relation']}:"
                        f"{row.get('kind') or ''}:{row['target']}"
                    ),
                    source=row["source"],
                    target=row["target"],
                    relation=row["relation"],
                    kind=row.get("kind"),
                )
                edges[edge.id] = edge
        if include_sources and memory_ids:
            source_rows = await self.client.execute_read(
                """
                MATCH (memory:Memory)-[:DERIVED_FROM]->(source:SourceMessage)
                WHERE memory.id IN $memory_ids
                RETURN memory.id AS memory_id, source
                ORDER BY source.timestamp, source.turn_index, source.id
                """,
                {"memory_ids": memory_ids},
            )
            for row in source_rows:
                source = SourceMessage.model_validate(_node(row, "source"))
                nodes[source.id] = GraphNode(
                    id=source.id,
                    node_type="source_message",
                    label=f"{source.role.value}: {source.content[:48]}",
                    data=source.model_dump(mode="json"),
                )
                edge = GraphEdge(
                    id=f"memory:{row['memory_id']}:DERIVED_FROM:{source.id}",
                    source=row["memory_id"],
                    target=source.id,
                    relation="DERIVED_FROM",
                )
                edges[edge.id] = edge
        return GraphSubgraph(nodes=list(nodes.values()), edges=list(edges.values()))

    async def create_correction_event(
        self, event: CorrectionEvent, target_memory_ids: list[str]
    ) -> CorrectionEvent:
        payload = event.model_dump(mode="json", exclude={"before", "after"})
        payload["before_json"] = _json(event.before)
        payload["after_json"] = _json(event.after)
        rows = await self.client.execute_write(
            """
            MATCH (target:Memory)
            WHERE target.id IN $target_ids
            WITH collect(target) AS targets
            WHERE size(targets) = size($target_ids)
            CREATE (event:CorrectionEvent $event)
            FOREACH (target IN targets | CREATE (event)-[:TARGETED]->(target))
            RETURN event
            """,
            {"event": payload, "target_ids": list(dict.fromkeys(target_memory_ids))},
        )
        if not rows:
            raise ValueError("Every correction target must exist")
        return self._correction_from_node(_node(rows[0], "event"))

    async def get_correction_event(self, event_id: str) -> CorrectionEvent | None:
        rows = await self.client.execute_read(
            "MATCH (event:CorrectionEvent {id: $id}) RETURN event",
            {"id": event_id},
        )
        return self._correction_from_node(_node(rows[0], "event")) if rows else None

    async def list_correction_events(self, memory_id: str) -> list[CorrectionEvent]:
        rows = await self.client.execute_read(
            """
            MATCH (event:CorrectionEvent)-[:TARGETED]->(:Memory {id: $memory_id})
            RETURN event ORDER BY event.timestamp, event.id
            """,
            {"memory_id": memory_id},
        )
        return [self._correction_from_node(_node(row, "event")) for row in rows]

    async def supersede_memory(self, old_memory_id: str, new_memory_id: str) -> None:
        rows = await self.client.execute_write(
            """
            MATCH (old:Memory {id: $old_id}), (new:Memory {id: $new_id})
            SET old.status = 'superseded',
                old.valid_to = coalesce(new.valid_from, new.created_at),
                old.updated_at = $updated_at,
                old.revision = old.revision + 1
            MERGE (new)-[:SUPERSEDES]->(old)
            MERGE (new)-[:CONTRADICTS]->(old)
            RETURN old.id AS id
            """,
            {
                "old_id": old_memory_id,
                "new_id": new_memory_id,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        if not rows:
            raise ValueError("Both memories must exist to record supersession")

    async def link_support(self, source_memory_id: str, target_memory_id: str) -> None:
        rows = await self.client.execute_write(
            """
            MATCH (source:Memory {id: $source_id}), (target:Memory {id: $target_id})
            MERGE (source)-[:SUPPORTS]->(target)
            RETURN source.id AS id
            """,
            {"source_id": source_memory_id, "target_id": target_memory_id},
        )
        if not rows:
            raise ValueError("Both memories must exist to record support")

    async def stats(self, backend_name: str = "scopegraph") -> MemoryStats:
        rows = await self.client.execute_read(
            """
            MATCH (n)
            WITH labels(n)[0] AS label, count(n) AS count
            RETURN collect({label: label, count: count}) AS counts
            """
        )
        counts = {item["label"]: item["count"] for item in rows[0]["counts"]} if rows else {}
        relationship_rows = await self.client.execute_read(
            "MATCH ()-[r]->() RETURN count(r) AS count"
        )
        return MemoryStats(
            backend_name=backend_name,
            scope_count=counts.get("Scope", 0),
            session_count=counts.get("Session", 0),
            source_message_count=counts.get("SourceMessage", 0),
            memory_count=counts.get("Memory", 0),
            relationship_count=relationship_rows[0]["count"] if relationship_rows else 0,
        )

    @staticmethod
    def _memory_from_row(row: dict[str, Any]) -> Memory:
        result = _node(row, "m")
        result["metadata"] = json.loads(result.pop("metadata_json", "{}"))
        result["source_ids"] = row.get("source_ids", [])
        return Memory.model_validate(result)

    @staticmethod
    def _correction_from_node(node: dict[str, Any]) -> CorrectionEvent:
        result = dict(node)
        result["before"] = json.loads(result.pop("before_json", "{}"))
        result["after"] = json.loads(result.pop("after_json", "{}"))
        return CorrectionEvent.model_validate(result)
