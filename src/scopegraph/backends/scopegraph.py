from datetime import UTC, datetime
from typing import Protocol

from scopegraph.llm.extraction import CandidateExtractor
from scopegraph.memory.base import MemorySystem
from scopegraph.memory.consolidator import ConsolidationRepository, Consolidator
from scopegraph.memory.corrections import CorrectionRepository, CorrectionService
from scopegraph.memory.promoter import PromotionPolicy
from scopegraph.memory.retriever import ScopeAwareRetriever
from scopegraph.models.correction import CorrectionRequest, CorrectionResult
from scopegraph.models.memory import Memory
from scopegraph.models.retrieval import IngestResult, MemoryStats, RetrievalResult
from scopegraph.models.scope import Scope, ScopeRef
from scopegraph.models.session import Session, SessionCreate, SessionInput
from scopegraph.models.source import SourceMessage, SourceMessageCreate


class WriteRepository(ConsolidationRepository, CorrectionRepository, Protocol):
    async def create_session(self, request: SessionCreate) -> Session: ...

    async def create_source_message(self, request: SourceMessageCreate) -> SourceMessage: ...

    async def list_source_messages(self, session_id: str) -> list[SourceMessage]: ...

    async def end_session(self, session_id: str, ended_at: datetime) -> Session | None: ...

    async def get_scope(self, scope_id: str) -> Scope | None: ...

    async def get_session(self, session_id: str) -> Session | None: ...

    async def get_global_scope(self) -> Scope | None: ...

    async def list_memories(
        self, *, scope_id: str | None = None, include_inactive: bool = False
    ) -> list[Memory]: ...

    async def stats(self, backend_name: str = "scopegraph") -> MemoryStats: ...


class ScopeGraphMemorySystem(MemorySystem):
    def __init__(
        self,
        repository: WriteRepository,
        extractor: CandidateExtractor,
        *,
        promotion_policy: PromotionPolicy | None = None,
        retriever: ScopeAwareRetriever | None = None,
    ) -> None:
        self.repository = repository
        self.extractor = extractor
        self.retriever = retriever
        self.corrections = CorrectionService(repository)
        self.consolidator = Consolidator(
            repository,
            promotion_policy or PromotionPolicy(),
        )

    async def reset(self) -> None:
        raise NotImplementedError("Reset requires an explicitly isolated experiment namespace")

    async def ingest_session(
        self, session: SessionInput, *, current_scope: ScopeRef | None
    ) -> IngestResult:
        scope_id = current_scope.id if current_scope else session.scope_id
        scope = await self.repository.get_scope(scope_id)
        if scope is None:
            raise ValueError(f"Scope {scope_id!r} does not exist")
        resolved_scope = current_scope or ScopeRef(
            id=scope.id, name=scope.name, scope_type=scope.scope_type
        )
        if session.scope_id != scope_id:
            raise ValueError("Session scope and explicit current scope must match")
        for message in session.messages:
            if message.session_id != session.id:
                raise ValueError("Every source message must reference the ingested session")

        await self.repository.create_session(
            SessionCreate(**session.model_dump(exclude={"messages"}))
        )
        for message in session.messages:
            await self.repository.create_source_message(message)
        return await self.consolidate_session(session.id, current_scope=resolved_scope)

    async def consolidate_session(
        self, session_id: str, *, current_scope: ScopeRef | None = None
    ) -> IngestResult:
        stored_session = await self.repository.get_session(session_id)
        if stored_session is None:
            raise ValueError(f"Session {session_id!r} does not exist")
        scope_id = current_scope.id if current_scope else stored_session.scope_id
        if stored_session.scope_id != scope_id:
            raise ValueError("Session scope and explicit current scope must match")
        scope = await self.repository.get_scope(scope_id)
        if scope is None:
            raise ValueError(f"Scope {scope_id!r} does not exist")
        resolved_scope = current_scope or ScopeRef(
            id=scope.id, name=scope.name, scope_type=scope.scope_type
        )
        stored_messages = await self.repository.list_source_messages(session_id)
        existing = await self.repository.list_memories(scope_id=scope_id, include_inactive=False)
        candidates = await self.extractor.extract(
            stored_messages,
            current_scope=resolved_scope,
            existing_memories=[memory.content for memory in existing],
        )
        global_scope = await self.repository.get_global_scope()
        outcome = await self.consolidator.consolidate(
            candidates,
            session_id=session_id,
            session_message_ids={message.id for message in stored_messages},
            source_timestamps={message.id: message.timestamp for message in stored_messages},
            current_scope=resolved_scope,
            global_scope_id=global_scope.id if global_scope else None,
        )
        await self.repository.end_session(session_id, stored_session.ended_at or datetime.now(UTC))
        return IngestResult(
            session_id=session_id,
            source_message_ids=[message.id for message in stored_messages],
            memory_ids=[memory.id for memory in outcome.memories],
            duplicate_count=outcome.duplicate_count,
            conflict_count=outcome.conflict_count,
            promoted_count=outcome.promoted_count,
        )

    async def retrieve(
        self,
        query: str,
        *,
        current_scope: ScopeRef | None,
        top_k: int,
        token_budget: int,
        now: datetime | None = None,
    ) -> RetrievalResult:
        if self.retriever is None:
            raise RuntimeError("Retrieval requires a configured embedding provider")
        return await self.retriever.retrieve(
            query,
            current_scope=current_scope,
            top_k=top_k,
            token_budget=token_budget,
            now=now,
        )

    async def apply_correction(self, correction: CorrectionRequest) -> CorrectionResult:
        return await self.corrections.apply(correction)

    async def stats(self) -> MemoryStats:
        return await self.repository.stats("scopegraph")
