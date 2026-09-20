import time
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from scopegraph.embeddings.base import EmbeddingProvider
from scopegraph.llm.extraction import CandidateExtractor
from scopegraph.memory.base import MemorySystem
from scopegraph.memory.normalizer import conflict_key, normalize_candidate, normalize_text
from scopegraph.memory.promoter import PromotionPolicy
from scopegraph.memory.provenance import validate_provenance
from scopegraph.memory.ranker import cosine_similarity, final_score, recency_score
from scopegraph.memory.retriever import RetrievalConfig
from scopegraph.memory.temporal import is_historical_query, temporal_score
from scopegraph.memory.traversal import MemoryNeighbor, bounded_traversal
from scopegraph.models.correction import CorrectionRequest, CorrectionResult
from scopegraph.models.memory import (
    Memory,
    MemoryCandidate,
    MemoryCreate,
    MemoryStatus,
    ScopeLevel,
)
from scopegraph.models.retrieval import (
    IngestResult,
    MemoryStats,
    RetrievalResult,
    RetrievedMemory,
    TraversalStep,
)
from scopegraph.models.scope import Scope, ScopeRef
from scopegraph.models.session import Session, SessionCreate, SessionInput
from scopegraph.models.source import SourceMessage, SourceMessageCreate
from scopegraph.observability.token_counting import pack_to_token_budget


class BaselineKind(StrEnum):
    VECTOR = "vector_memory"
    FLAT_GRAPH = "flat_graph"
    TWO_LEVEL_GRAPH = "two_level_graph"


class BaselineRepository(Protocol):
    async def create_session(self, request: SessionCreate) -> Session: ...

    async def create_source_message(self, request: SourceMessageCreate) -> SourceMessage: ...

    async def list_source_messages(self, session_id: str) -> list[SourceMessage]: ...

    async def end_session(self, session_id: str, ended_at: datetime) -> Session | None: ...

    async def get_scope(self, scope_id: str) -> Scope | None: ...

    async def get_global_scope(self) -> Scope | None: ...

    async def list_memories(
        self, *, scope_id: str | None = None, include_inactive: bool = False
    ) -> list[Memory]: ...

    async def create_memory(self, request: MemoryCreate) -> Memory: ...

    async def supersede_memory(self, old_memory_id: str, new_memory_id: str) -> None: ...

    async def set_memory_embedding(
        self, memory_id: str, embedding: list[float], model_name: str
    ) -> None: ...

    async def get_memory_neighbors(self, memory_ids: list[str]) -> list[MemoryNeighbor]: ...

    async def stats(self, backend_name: str = "scopegraph") -> MemoryStats: ...


class BaselineMemorySystem(MemorySystem):
    """Shared, controlled implementation for the three Phase 4 baselines."""

    def __init__(
        self,
        repository: BaselineRepository,
        extractor: CandidateExtractor,
        embedder: EmbeddingProvider,
        *,
        kind: BaselineKind,
        promotion_policy: PromotionPolicy | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        self.repository = repository
        self.extractor = extractor
        self.embedder = embedder
        self.kind = kind
        self.promotion_policy = promotion_policy or PromotionPolicy()
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.retrieval_config.weights.validate()

    async def reset(self) -> None:
        raise NotImplementedError("Reset requires an explicitly isolated experiment namespace")

    async def ingest_session(
        self, session: SessionInput, *, current_scope: ScopeRef | None
    ) -> IngestResult:
        scope_id = current_scope.id if current_scope else session.scope_id
        scope = await self.repository.get_scope(scope_id)
        if scope is None:
            raise ValueError(f"Scope {scope_id!r} does not exist")
        if session.scope_id != scope_id:
            raise ValueError("Session scope and explicit current scope must match")
        if any(message.session_id != session.id for message in session.messages):
            raise ValueError("Every source message must reference the ingested session")
        resolved_scope = current_scope or ScopeRef(
            id=scope.id, name=scope.name, scope_type=scope.scope_type
        )

        await self.repository.create_session(
            SessionCreate(**session.model_dump(exclude={"messages"}))
        )
        for message in session.messages:
            await self.repository.create_source_message(message)
        stored_messages = await self.repository.list_source_messages(session.id)
        existing = await self.repository.list_memories(include_inactive=False)
        candidates = await self.extractor.extract(
            stored_messages,
            current_scope=resolved_scope,
            existing_memories=[memory.content for memory in existing],
        )
        global_scope = await self.repository.get_global_scope()
        created: list[Memory] = []
        conflict_count = 0
        valid_source_ids = {message.id for message in stored_messages}
        for raw_candidate in candidates:
            validate_provenance(raw_candidate, valid_source_ids)
            candidate = normalize_candidate(raw_candidate)
            if candidate.valid_from is None:
                timestamps = [
                    message.timestamp
                    for message in stored_messages
                    if message.id in candidate.source_message_ids
                ]
                if timestamps:
                    candidate = candidate.model_copy(update={"valid_from": min(timestamps)})
            target_scope_id, scope_level = self._placement(
                candidate,
                session_scope_id=scope_id,
                global_scope_id=global_scope.id if global_scope else None,
            )
            conflicts = self._conflicts(
                candidate,
                existing,
                scope_level=scope_level,
                session_id=session.id,
            )
            memory = await self.repository.create_memory(
                MemoryCreate(
                    content=candidate.content,
                    memory_type=candidate.memory_type,
                    scope_level=scope_level,
                    scope_id=target_scope_id,
                    confidence=candidate.confidence,
                    valid_from=candidate.valid_from,
                    valid_to=candidate.valid_to,
                    source_ids=candidate.source_message_ids,
                    metadata={
                        "subject": candidate.subject,
                        "predicate": candidate.predicate,
                        "object": candidate.object,
                        "conflict_key": conflict_key(candidate),
                        "durability": candidate.durability,
                        "session_id": session.id,
                        "original_scope_id": scope_id,
                        "baseline": self.kind.value,
                    },
                )
            )
            for old_memory in conflicts:
                await self.repository.supersede_memory(old_memory.id, memory.id)
                old_memory.status = MemoryStatus.SUPERSEDED
            conflict_count += len(conflicts)
            created.append(memory)
            existing.append(memory)
        await self.repository.end_session(session.id, session.ended_at or datetime.now(UTC))
        return IngestResult(
            session_id=session.id,
            source_message_ids=[message.id for message in stored_messages],
            memory_ids=[memory.id for memory in created],
            conflict_count=conflict_count,
        )

    def _placement(
        self,
        candidate: MemoryCandidate,
        *,
        session_scope_id: str,
        global_scope_id: str | None,
    ) -> tuple[str, ScopeLevel]:
        if self.kind is not BaselineKind.TWO_LEVEL_GRAPH:
            # Required by the shared schema, but ignored as a retrieval boundary.
            return session_scope_id, ScopeLevel.SCOPE
        durable = candidate.proposed_scope_level == "global" or (
            self.promotion_policy.should_promote_session_to_scope(candidate)
        )
        if not durable:
            return session_scope_id, ScopeLevel.SESSION
        if global_scope_id is None:
            raise ValueError("TwoLevelGraphMemory requires a global root scope")
        return global_scope_id, ScopeLevel.GLOBAL

    def _conflicts(
        self,
        candidate: MemoryCandidate,
        existing: list[Memory],
        *,
        scope_level: ScopeLevel,
        session_id: str,
    ) -> list[Memory]:
        if self.kind is BaselineKind.VECTOR:
            return []
        key = conflict_key(candidate)
        if key is None or candidate.object is None:
            return []
        conflicts: list[Memory] = []
        for memory in existing:
            if memory.status is not MemoryStatus.ACTIVE:
                continue
            if self.kind is BaselineKind.TWO_LEVEL_GRAPH:
                if memory.scope_level is not scope_level:
                    continue
                if scope_level is ScopeLevel.SESSION and (
                    memory.metadata.get("session_id") != session_id
                ):
                    continue
            if memory.metadata.get("conflict_key") != key:
                continue
            old_object = memory.metadata.get("object")
            if isinstance(old_object, str) and normalize_text(old_object) != candidate.object:
                conflicts.append(memory)
        return conflicts

    async def retrieve(
        self,
        query: str,
        *,
        current_scope: ScopeRef | None,
        top_k: int,
        token_budget: int,
        now: datetime | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        effective_now = now or datetime.now(UTC)
        historical = is_historical_query(
            query, self.retrieval_config.historical_query_terms
        )
        memories = await self.repository.list_memories(include_inactive=historical)
        candidates = [
            memory
            for memory in memories
            if self._eligible(memory, current_scope, historical=historical)
        ]
        query_vector = (await self.embedder.embed([query]))[0]
        await self._ensure_embeddings(candidates)
        semantic = {
            memory.id: cosine_similarity(query_vector, memory.embedding or [])
            for memory in candidates
        }
        anchor_limit = min(len(candidates), max(top_k * 3, top_k))
        anchors = sorted(
            candidates, key=lambda memory: semantic[memory.id], reverse=True
        )[:anchor_limit]

        expanded: dict[str, tuple[Memory, int]] = {}
        traversal_steps: list[TraversalStep] = []
        if self.kind is not BaselineKind.VECTOR:
            expanded, traversal_steps = await bounded_traversal(
                self.repository,
                [memory.id for memory in anchors],
                allowed_scope_ids={memory.scope_id for memory in candidates},
                max_hops=self.retrieval_config.max_graph_hops,
                max_expanded_nodes=self.retrieval_config.max_expanded_nodes,
                max_time_ms=self.retrieval_config.max_traversal_time_ms,
            )
        expanded_memories = [
            memory
            for memory, _ in expanded.values()
            if self._eligible(memory, current_scope, historical=historical)
        ]
        await self._ensure_embeddings(expanded_memories)
        for memory in expanded_memories:
            semantic[memory.id] = cosine_similarity(query_vector, memory.embedding or [])

        all_memories = {memory.id: memory for memory in candidates}
        all_memories.update({memory.id: memory for memory in expanded_memories})
        anchor_ids = {memory.id for memory in anchors}
        step_by_id = {step.to_id: step for step in traversal_steps}
        ranked: list[tuple[RetrievedMemory, TraversalStep]] = []
        for memory in all_memories.values():
            temporal = temporal_score(memory, now=effective_now, historical=historical)
            if temporal == 0:
                continue
            depth = expanded.get(memory.id, (memory, 0))[1]
            graph_score = 0.0
            if self.kind is not BaselineKind.VECTOR:
                graph_score = 1.0 if memory.id in anchor_ids else 1.0 / (depth + 1)
            scope_score = self._scope_score(memory, current_scope)
            score = semantic[memory.id]
            if self.kind is not BaselineKind.VECTOR:
                score = final_score(
                    semantic=semantic[memory.id],
                    scope=scope_score,
                    temporal=temporal,
                    confidence=memory.confidence,
                    graph=graph_score,
                    recency=recency_score(memory, now=effective_now),
                    weights=self.retrieval_config.weights,
                )
            item = RetrievedMemory(
                memory_id=memory.id,
                content=memory.content,
                score=score,
                scope_id=memory.scope_id,
                scope_level=memory.scope_level,
                status=memory.status,
                source_ids=memory.source_ids,
                valid_from=memory.valid_from,
                valid_to=memory.valid_to,
                semantic_score=semantic[memory.id],
                scope_score=scope_score,
                temporal_score=temporal,
                graph_score=graph_score,
                confidence=memory.confidence,
            )
            base_step = step_by_id.get(memory.id)
            trace = TraversalStep(
                from_id=base_step.from_id if base_step else "flat-semantic-store",
                to_id=memory.id,
                relation=base_step.relation if base_step else "SEMANTIC_ANCHOR",
                depth=base_step.depth if base_step else 0,
                path=base_step.path if base_step else [memory.id],
                semantic_score=semantic[memory.id],
                scope_score=scope_score,
                temporal_score=temporal,
                graph_score=graph_score,
                final_score=score,
                reason=self._selection_reason(memory, current_scope),
            )
            ranked.append((item, trace))

        ranked.sort(key=lambda pair: pair[0].score, reverse=True)
        packed, token_count = pack_to_token_budget(
            [item for item, _ in ranked[:top_k]], token_budget
        )
        selected_ids = {item.memory_id for item in packed}
        elapsed = (time.perf_counter() - started) * 1000
        return RetrievalResult(
            items=packed,
            retrieval_latency_ms=elapsed,
            token_count=token_count,
            trace=[trace for item, trace in ranked if item.memory_id in selected_ids],
            backend_name=self.kind.value,
        )

    def _eligible(
        self, memory: Memory, current_scope: ScopeRef | None, *, historical: bool
    ) -> bool:
        if memory.status is MemoryStatus.TOMBSTONED:
            return False
        if not historical and memory.status is not MemoryStatus.ACTIVE:
            return False
        if self.kind is not BaselineKind.TWO_LEVEL_GRAPH:
            return True
        if memory.scope_level is ScopeLevel.GLOBAL:
            return True
        return bool(
            memory.scope_level is ScopeLevel.SESSION
            and current_scope is not None
            and current_scope.session_id is not None
            and memory.metadata.get("session_id") == current_scope.session_id
        )

    def _scope_score(self, memory: Memory, current_scope: ScopeRef | None) -> float:
        if self.kind is not BaselineKind.TWO_LEVEL_GRAPH:
            return 0.0
        if memory.scope_level is ScopeLevel.SESSION and current_scope is not None:
            if memory.metadata.get("session_id") == current_scope.session_id:
                return self.retrieval_config.current_session_score
        return self.retrieval_config.global_scope_score

    def _selection_reason(self, memory: Memory, current_scope: ScopeRef | None) -> str:
        if self.kind is BaselineKind.VECTOR:
            return "flat semantic similarity; scope hierarchy disabled"
        if self.kind is BaselineKind.FLAT_GRAPH:
            return "hybrid semantic/graph retrieval; scope hierarchy disabled"
        if memory.scope_level is ScopeLevel.SESSION and current_scope is not None:
            return "matching current session"
        return "global memory in two-level hierarchy"

    async def _ensure_embeddings(self, memories: list[Memory]) -> None:
        missing = [
            memory
            for memory in memories
            if memory.embedding is None or memory.embedding_model != self.embedder.model_name
        ]
        if not missing:
            return
        vectors = await self.embedder.embed([memory.content for memory in missing])
        if len(vectors) != len(missing):
            raise ValueError("Embedding provider returned the wrong number of vectors")
        for memory, vector in zip(missing, vectors, strict=True):
            memory.embedding = vector
            memory.embedding_model = self.embedder.model_name
            await self.repository.set_memory_embedding(memory.id, vector, self.embedder.model_name)

    async def apply_correction(self, correction: CorrectionRequest) -> CorrectionResult:
        del correction
        raise NotImplementedError("Correction workflows are implemented in Phase 5")

    async def stats(self) -> MemoryStats:
        return await self.repository.stats(self.kind.value)
