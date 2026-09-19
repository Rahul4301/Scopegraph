from datetime import UTC, datetime

from scopegraph.models.memory import Memory, MemoryCreate, MemoryUpdate
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

    async def create_source_message(self, request: SourceMessageCreate) -> SourceMessage:
        if request.session_id not in self.sessions:
            raise ValueError(f"Session {request.session_id!r} does not exist")
        message = SourceMessage.model_validate(request.model_dump())
        self.messages[message.id] = message
        return message

    async def get_source_message(self, message_id: str) -> SourceMessage | None:
        return self.messages.get(message_id)

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

    async def update_memory(self, memory_id: str, update: MemoryUpdate) -> Memory | None:
        current = self.memories.get(memory_id)
        if current is None:
            return None
        changes = update.model_dump(exclude_unset=True)
        changes.update(updated_at=datetime.now(UTC), revision=current.revision + 1)
        memory = current.model_copy(update=changes)
        self.memories[memory_id] = memory
        return memory
