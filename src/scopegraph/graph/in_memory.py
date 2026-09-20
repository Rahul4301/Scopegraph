from datetime import UTC, datetime

from scopegraph.memory.traversal import MemoryNeighbor
from scopegraph.models.correction import (
    CorrectionEvent,
    CorrectionRelation,
    SupportDependency,
)
from scopegraph.models.graph import GraphEdge, GraphNode, GraphSubgraph
from scopegraph.models.memory import (
    Memory,
    MemoryCreate,
    MemoryStatus,
    MemoryUpdate,
    ScopeLevel,
)
from scopegraph.models.relationship import RelationKind
from scopegraph.models.retrieval import MemoryStats
from scopegraph.models.scope import Scope, ScopeCreate, ScopeUpdate
from scopegraph.models.session import Session, SessionCreate
from scopegraph.models.source import SourceMessage, SourceMessageCreate


class InMemoryMemoryRepository:
    """Deterministic repository used for unit tests and offline development."""

    def __init__(self) -> None:
        self.scopes: dict[str, Scope] = {}
        self.sessions: dict[str, Session] = {}
        self.messages: dict[str, SourceMessage] = {}
        self.memories: dict[str, Memory] = {}
        self.supersedes: set[tuple[str, str]] = set()
        self.supports: set[tuple[str, str]] = set()
        self.same_as: set[tuple[str, str]] = set()
        self.contradicts: set[tuple[str, str]] = set()
        self.relates_to: set[tuple[str, str, RelationKind]] = set()
        self.correction_events: dict[str, CorrectionEvent] = {}
        self.correction_targets: dict[str, set[str]] = {}

    async def create_scope(self, request: ScopeCreate) -> Scope:
        if request.id in self.scopes:
            raise ValueError(f"Scope {request.id!r} already exists")
        if request.scope_type.value == "global" and any(
            item.scope_type.value == "global" and not item.archived
            for item in self.scopes.values()
        ):
            raise ValueError("An active global root scope already exists")
        if request.parent_scope_id and request.parent_scope_id not in self.scopes:
            raise ValueError(f"Parent scope {request.parent_scope_id!r} does not exist")
        scope = Scope.model_validate(request.model_dump())
        self.scopes[scope.id] = scope
        return scope

    async def get_scope(self, scope_id: str) -> Scope | None:
        return self.scopes.get(scope_id)

    async def list_scopes(self, *, include_archived: bool = False) -> list[Scope]:
        return [
            item
            for item in self.scopes.values()
            if include_archived or not item.archived
        ]

    async def get_global_scope(self) -> Scope | None:
        return next(
            (
                item
                for item in self.scopes.values()
                if item.scope_type.value == "global" and not item.archived
            ),
            None,
        )

    async def update_scope(self, scope_id: str, update: ScopeUpdate) -> Scope | None:
        current = self.scopes.get(scope_id)
        if current is None:
            return None
        if (
            "parent_scope_id" in update.model_fields_set
            and update.parent_scope_id is not None
            and update.parent_scope_id not in self.scopes
        ):
            raise ValueError(f"Parent scope {update.parent_scope_id!r} does not exist")
        scope = current.model_copy(update=update.model_dump(exclude_unset=True))
        self.scopes[scope_id] = scope
        return scope

    async def create_session(self, request: SessionCreate) -> Session:
        if request.scope_id not in self.scopes:
            raise ValueError(f"Scope {request.scope_id!r} does not exist")
        session = Session.model_validate(request.model_dump())
        self.sessions[session.id] = session
        return session

    async def get_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    async def end_session(self, session_id: str, ended_at: datetime) -> Session | None:
        current = self.sessions.get(session_id)
        if current is None:
            return None
        session = current.model_copy(update={"ended_at": ended_at})
        self.sessions[session_id] = session
        return session

    async def create_source_message(self, request: SourceMessageCreate) -> SourceMessage:
        if request.session_id not in self.sessions:
            raise ValueError(f"Session {request.session_id!r} does not exist")
        message = SourceMessage.model_validate(request.model_dump())
        self.messages[message.id] = message
        return message

    async def get_source_message(self, message_id: str) -> SourceMessage | None:
        return self.messages.get(message_id)

    async def list_source_messages(self, session_id: str) -> list[SourceMessage]:
        return sorted(
            (item for item in self.messages.values() if item.session_id == session_id),
            key=lambda item: (item.turn_index, item.timestamp),
        )

    async def get_source_messages_by_ids(
        self, message_ids: list[str]
    ) -> list[SourceMessage]:
        requested = set(message_ids)
        return sorted(
            (message for key, message in self.messages.items() if key in requested),
            key=lambda message: (message.timestamp, message.turn_index, message.id),
        )

    async def create_memory(self, request: MemoryCreate) -> Memory:
        if request.scope_id not in self.scopes:
            raise ValueError(f"Scope {request.scope_id!r} does not exist")
        missing = set(request.source_ids) - self.messages.keys()
        if missing:
            raise ValueError(f"Source messages do not exist: {sorted(missing)}")
        memory = Memory.model_validate(request.model_dump())
        self.memories[memory.id] = memory
        return memory

    async def get_memory(self, memory_id: str) -> Memory | None:
        return self.memories.get(memory_id)

    async def list_memories(
        self, *, scope_id: str | None = None, include_inactive: bool = False
    ) -> list[Memory]:
        return [
            item
            for item in self.memories.values()
            if (scope_id is None or item.scope_id == scope_id)
            and (include_inactive or item.status.value == "active")
        ]

    async def set_memory_embedding(
        self, memory_id: str, embedding: list[float], model_name: str
    ) -> None:
        current = self.memories.get(memory_id)
        if current is None:
            raise ValueError(f"Memory {memory_id!r} does not exist")
        self.memories[memory_id] = current.model_copy(
            update={"embedding": embedding, "embedding_model": model_name}
        )

    async def get_memory_neighbors(self, memory_ids: list[str]) -> list[MemoryNeighbor]:
        requested = set(memory_ids)
        neighbors: list[MemoryNeighbor] = []
        relationships = [
            *((source, target, "SUPERSEDES") for source, target in self.supersedes),
            *((source, target, "SUPPORTS") for source, target in self.supports),
            *((source, target, "SAME_AS") for source, target in self.same_as),
            *((source, target, "CONTRADICTS") for source, target in self.contradicts),
            *(
                (source, target, f"RELATES_TO:{kind.value}")
                for source, target, kind in self.relates_to
            ),
        ]
        for source, target, relation in relationships:
            if source in requested and target in self.memories:
                neighbors.append(MemoryNeighbor(source, self.memories[target], relation))
            if target in requested and source in self.memories:
                neighbors.append(MemoryNeighbor(target, self.memories[source], relation))
        return neighbors

    async def update_memory(self, memory_id: str, update: MemoryUpdate) -> Memory | None:
        current = self.memories.get(memory_id)
        if current is None:
            return None
        changes = update.model_dump(exclude_unset=True)
        changes.update(updated_at=datetime.now(UTC), revision=current.revision + 1)
        memory = current.model_copy(update=changes)
        self.memories[memory_id] = memory
        return memory

    async def move_memory(
        self, memory_id: str, scope_id: str, scope_level: ScopeLevel
    ) -> Memory | None:
        if scope_id not in self.scopes:
            raise ValueError(f"Scope {scope_id!r} does not exist")
        return await self.update_memory(
            memory_id, MemoryUpdate(scope_id=scope_id, scope_level=scope_level)
        )

    async def merge_memories(
        self, source_memory_id: str, target_memory_id: str
    ) -> tuple[Memory, Memory]:
        source = self.memories.get(source_memory_id)
        target = self.memories.get(target_memory_id)
        if source is None or target is None:
            raise ValueError("Both memories must exist to merge")
        now = datetime.now(UTC)
        updated_target = target.model_copy(
            update={
                "source_ids": list(dict.fromkeys([*target.source_ids, *source.source_ids])),
                "updated_at": now,
                "revision": target.revision + 1,
            }
        )
        updated_source = source.model_copy(
            update={
                "status": MemoryStatus.TOMBSTONED,
                "updated_at": now,
                "revision": source.revision + 1,
            }
        )
        self.memories[target_memory_id] = updated_target
        self.memories[source_memory_id] = updated_source
        self.same_as.add((source_memory_id, target_memory_id))
        return updated_source, updated_target

    async def add_memory_relation(
        self,
        source_memory_id: str,
        target_memory_id: str,
        relation: CorrectionRelation,
        kind: RelationKind | None = None,
    ) -> Memory:
        source = self.memories.get(source_memory_id)
        if source is None or target_memory_id not in self.memories:
            raise ValueError("Both memories must exist to add a relation")
        if source_memory_id == target_memory_id:
            raise ValueError("A memory cannot relate to itself")
        pair = (source_memory_id, target_memory_id)
        if relation is CorrectionRelation.SUPPORTS:
            self.supports.add(pair)
        elif relation is CorrectionRelation.SAME_AS:
            self.same_as.add(pair)
        elif relation is CorrectionRelation.CONTRADICTS:
            self.contradicts.add(pair)
        elif relation is CorrectionRelation.RELATES_TO and kind is not None:
            self.relates_to.add((*pair, kind))
        else:
            raise ValueError("RELATES_TO requires an allowlisted kind")
        updated = source.model_copy(
            update={"updated_at": datetime.now(UTC), "revision": source.revision + 1}
        )
        self.memories[source_memory_id] = updated
        return updated

    async def remove_memory_relation(
        self,
        source_memory_id: str,
        target_memory_id: str,
        relation: CorrectionRelation,
        kind: RelationKind | None = None,
    ) -> Memory:
        source = self.memories.get(source_memory_id)
        if source is None or target_memory_id not in self.memories:
            raise ValueError("Both memories must exist to remove a relation")
        pair = (source_memory_id, target_memory_id)
        removed = False
        if relation is CorrectionRelation.SUPPORTS and pair in self.supports:
            self.supports.remove(pair)
            removed = True
        elif relation is CorrectionRelation.SAME_AS and pair in self.same_as:
            self.same_as.remove(pair)
            removed = True
        elif relation is CorrectionRelation.CONTRADICTS and pair in self.contradicts:
            self.contradicts.remove(pair)
            removed = True
        elif relation is CorrectionRelation.RELATES_TO and kind is not None:
            semantic = (*pair, kind)
            if semantic in self.relates_to:
                self.relates_to.remove(semantic)
                removed = True
        if not removed:
            raise ValueError("The requested memory relation does not exist")
        updated = source.model_copy(
            update={"updated_at": datetime.now(UTC), "revision": source.revision + 1}
        )
        self.memories[source_memory_id] = updated
        return updated

    async def get_support_dependents(self, memory_id: str) -> list[SupportDependency]:
        dependencies: list[SupportDependency] = []
        dependent_ids = {target for source, target in self.supports if source == memory_id}
        for dependent_id in sorted(dependent_ids):
            dependent = self.memories.get(dependent_id)
            if dependent is None:
                continue
            other_support_ids = sorted(
                source
                for source, target in self.supports
                if target == dependent_id
                and source != memory_id
                and source in self.memories
                and self.memories[source].status is MemoryStatus.ACTIVE
            )
            dependencies.append(
                SupportDependency(
                    memory=dependent,
                    other_active_support_ids=other_support_ids,
                )
            )
        return dependencies

    async def get_subgraph(
        self,
        *,
        scope_id: str | None = None,
        memory_id: str | None = None,
        include_inactive: bool = True,
        include_sources: bool = False,
        limit: int = 200,
    ) -> GraphSubgraph:
        scopes = list(self.scopes.values())
        memories = [
            memory
            for memory in self.memories.values()
            if (scope_id is None or memory.scope_id == scope_id or memory_id is not None)
            and (include_inactive or memory.status is MemoryStatus.ACTIVE)
        ]
        if memory_id is not None:
            neighbor_ids = {memory_id}
            for source, target in (
                self.supersedes | self.supports | self.same_as | self.contradicts
            ):
                if source == memory_id:
                    neighbor_ids.add(target)
                if target == memory_id:
                    neighbor_ids.add(source)
            for source, target, _ in self.relates_to:
                if source == memory_id:
                    neighbor_ids.add(target)
                if target == memory_id:
                    neighbor_ids.add(source)
            memories = [memory for memory in memories if memory.id in neighbor_ids]
        memories = sorted(memories, key=lambda memory: (memory.created_at, memory.id))[:limit]
        memory_ids = {memory.id for memory in memories}
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
        relations = [
            *((source, target, "SUPERSEDES", None) for source, target in self.supersedes),
            *((source, target, "SUPPORTS", None) for source, target in self.supports),
            *((source, target, "SAME_AS", None) for source, target in self.same_as),
            *((source, target, "CONTRADICTS", None) for source, target in self.contradicts),
            *(
                (source, target, "RELATES_TO", kind.value)
                for source, target, kind in self.relates_to
            ),
        ]
        for source, target, relation, kind in relations:
            if source not in memory_ids or target not in memory_ids:
                continue
            edge = GraphEdge(
                id=f"memory:{source}:{relation}:{kind or ''}:{target}",
                source=source,
                target=target,
                relation=relation,
                kind=kind,
            )
            edges[edge.id] = edge
        if include_sources:
            for memory in memories:
                for source_id in memory.source_ids:
                    source_message = self.messages.get(source_id)
                    if source_message is None:
                        continue
                    nodes[source_message.id] = GraphNode(
                        id=source_message.id,
                        node_type="source_message",
                        label=(
                            f"{source_message.role.value}: "
                            f"{source_message.content[:48]}"
                        ),
                        data=source_message.model_dump(mode="json"),
                    )
                    edge = GraphEdge(
                        id=f"memory:{memory.id}:DERIVED_FROM:{source_message.id}",
                        source=memory.id,
                        target=source_message.id,
                        relation="DERIVED_FROM",
                    )
                    edges[edge.id] = edge
        return GraphSubgraph(nodes=list(nodes.values()), edges=list(edges.values()))

    async def create_correction_event(
        self, event: CorrectionEvent, target_memory_ids: list[str]
    ) -> CorrectionEvent:
        missing = set(target_memory_ids) - self.memories.keys()
        if missing:
            raise ValueError(f"Correction targets do not exist: {sorted(missing)}")
        if event.id in self.correction_events:
            raise ValueError(f"Correction event {event.id!r} already exists")
        self.correction_events[event.id] = event
        self.correction_targets[event.id] = set(target_memory_ids)
        return event

    async def get_correction_event(self, event_id: str) -> CorrectionEvent | None:
        return self.correction_events.get(event_id)

    async def list_correction_events(self, memory_id: str) -> list[CorrectionEvent]:
        return sorted(
            (
                event
                for event_id, event in self.correction_events.items()
                if memory_id in self.correction_targets[event_id]
            ),
            key=lambda event: (event.timestamp, event.id),
        )

    async def supersede_memory(self, old_memory_id: str, new_memory_id: str) -> None:
        old = self.memories.get(old_memory_id)
        new = self.memories.get(new_memory_id)
        if old is None or new is None:
            raise ValueError("Both memories must exist to record supersession")
        self.memories[old_memory_id] = old.model_copy(
            update={
                "status": MemoryStatus.SUPERSEDED,
                "valid_to": new.valid_from or new.created_at,
                "updated_at": datetime.now(UTC),
                "revision": old.revision + 1,
            }
        )
        self.supersedes.add((new_memory_id, old_memory_id))
        self.contradicts.add((new_memory_id, old_memory_id))

    async def link_support(self, source_memory_id: str, target_memory_id: str) -> None:
        if source_memory_id not in self.memories or target_memory_id not in self.memories:
            raise ValueError("Both memories must exist to record support")
        self.supports.add((source_memory_id, target_memory_id))

    async def stats(self, backend_name: str = "scopegraph") -> MemoryStats:
        hierarchy_edges = sum(1 for item in self.scopes.values() if item.parent_scope_id)
        provenance_edges = sum(len(item.source_ids) for item in self.memories.values())
        return MemoryStats(
            backend_name=backend_name,
            scope_count=len(self.scopes),
            session_count=len(self.sessions),
            source_message_count=len(self.messages),
            memory_count=len(self.memories),
            relationship_count=(
                hierarchy_edges
                + len(self.sessions)
                + len(self.messages)
                + len(self.memories)
                + provenance_edges
                + len(self.supersedes)
                + len(self.supports)
                + len(self.same_as)
                + len(self.contradicts)
                + len(self.relates_to)
                + sum(len(targets) for targets in self.correction_targets.values())
            ),
        )
