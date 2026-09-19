from datetime import UTC, datetime

from scopegraph.memory.traversal import MemoryNeighbor
from scopegraph.models.memory import Memory, MemoryCreate, MemoryStatus, MemoryUpdate
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
            ),
        )
