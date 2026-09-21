import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from heapq import nlargest
from typing import Any, Protocol

from scopegraph.embeddings.base import EmbeddingProvider
from scopegraph.memory.ranker import (
    RankingWeights,
    cosine_similarity,
    final_score,
    recency_score,
)
from scopegraph.memory.temporal import is_historical_query, temporal_score
from scopegraph.memory.traversal import MemoryNeighbor, bounded_traversal
from scopegraph.models.memory import Memory, MemoryStatus, ScopeLevel
from scopegraph.models.retrieval import RetrievalResult, RetrievedMemory, TraversalStep
from scopegraph.models.scope import Scope, ScopeRef, ScopeType
from scopegraph.observability.token_counting import pack_to_token_budget


class RetrievalRepository(Protocol):
    async def list_scopes(self, *, include_archived: bool = False) -> list[Scope]: ...

    async def get_scope(self, scope_id: str) -> Scope | None: ...

    async def get_global_scope(self) -> Scope | None: ...

    async def list_memories(
        self, *, scope_id: str | None = None, include_inactive: bool = False
    ) -> list[Memory]: ...

    async def list_retrieval_memories(
        self, *, scope_ids: set[str], session_id: str | None,
        historical: bool, now: datetime,
    ) -> list[Memory]: ...

    async def set_memory_embeddings(
        self, values: list[tuple[str, str, list[float]]], model_name: str
    ) -> None: ...

    async def set_memory_embedding(
        self, memory_id: str, embedding: list[float], model_name: str
    ) -> None: ...

    async def get_memory_neighbors(
        self, memory_ids: list[str], *, eligible_ids: set[str] | None = None,
        limit: int | None = None,
    ) -> list[MemoryNeighbor]: ...


@dataclass(frozen=True)
class RetrievalConfig:
    weights: RankingWeights = RankingWeights()
    max_graph_hops: int = 2
    max_expanded_nodes: int = 40
    max_traversal_time_ms: float = 100.0
    historical_query_terms: tuple[str, ...] = (
        "before",
        "previous",
        "previously",
        "used to",
        "history",
        "historical",
        "formerly",
    )
    current_session_score: float = 1.0
    current_scope_score: float = 0.9
    parent_scope_score: float = 0.7
    explicit_named_scope_score: float = 0.85
    global_scope_score: float = 0.5

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RetrievalConfig":
        scope = config.get("scope", {})
        return cls(
            weights=RankingWeights.from_config(config),
            max_graph_hops=int(config.get("max_graph_hops", 2)),
            max_expanded_nodes=int(config.get("max_expanded_nodes", 40)),
            max_traversal_time_ms=float(config.get("max_traversal_time_ms", 100.0)),
            historical_query_terms=tuple(config.get(
                "historical_query_terms", cls().historical_query_terms
            )),
            current_session_score=float(scope.get("current_session", 1.0)),
            current_scope_score=float(scope.get("current_scope", 0.9)),
            parent_scope_score=float(scope.get("parent_scope", 0.7)),
            explicit_named_scope_score=float(scope.get("explicit_named_scope", 0.85)),
            global_scope_score=float(scope.get("global", 0.5)),
        )


@dataclass(frozen=True)
class ScopeAccess:
    score: float
    reason: str


class ScopeAwareRetriever:
    def __init__(
        self,
        repository: RetrievalRepository,
        embedder: EmbeddingProvider,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.repository = repository
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self.config.weights.validate()

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
        access = await self._scope_access(query, current_scope)
        historical = is_historical_query(query, self.config.historical_query_terms)
        candidates = await self._eligible_memories(
            access, current_scope=current_scope, historical=historical, now=effective_now
        )
        if not candidates:
            return RetrievalResult(items=[], trace=[], token_count=0, backend_name="scopegraph",
                                   retrieval_latency_ms=(time.perf_counter() - started) * 1000)
        query_vector = (await self.embedder.embed([query]))[0]
        await self._ensure_embeddings(candidates)
        semantic = {
            memory.id: cosine_similarity(query_vector, memory.embedding or [])
            for memory in candidates
        }
        anchor_limit = min(len(candidates), max(top_k * 3, top_k))
        anchors = nlargest(
            anchor_limit,
            candidates,
            key=lambda memory: (
                0.65 * semantic[memory.id] + 0.35 * access[memory.scope_id].score
            ),
        )
        expanded, traversal_steps = await bounded_traversal(
            self.repository,
            [memory.id for memory in anchors],
            allowed_scope_ids=set(access),
            max_hops=self.config.max_graph_hops,
            max_expanded_nodes=self.config.max_expanded_nodes,
            max_time_ms=self.config.max_traversal_time_ms,
            eligible_ids={memory.id for memory in candidates},
        )
        eligible_ids = {memory.id for memory in candidates}
        expanded = {key: value for key, value in expanded.items() if key in eligible_ids}
        expanded_memories = [value[0] for value in expanded.values()]
        all_memories = {memory.id: memory for memory in anchors}
        all_memories.update({memory.id: memory for memory in expanded_memories})
        anchor_ids = {memory.id for memory in anchors}
        step_by_id = {step.to_id: step for step in traversal_steps}
        ranked: list[tuple[RetrievedMemory, TraversalStep]] = []
        for memory in all_memories.values():
            temporal = temporal_score(memory, now=effective_now, historical=historical)
            if temporal == 0:
                continue
            depth = expanded.get(memory.id, (memory, 0))[1]
            graph = 1.0 if memory.id in anchor_ids else 1.0 / (depth + 1)
            scope_access = access[memory.scope_id]
            recency = recency_score(memory, now=effective_now)
            score = final_score(
                semantic=semantic[memory.id],
                scope=self._memory_scope_score(
                    memory, current_scope, scope_access.score
                ),
                temporal=temporal,
                confidence=memory.confidence,
                graph=graph,
                recency=recency,
                weights=self.config.weights,
            )
            retrieved = RetrievedMemory(
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
                scope_score=self._memory_scope_score(
                    memory, current_scope, scope_access.score
                ),
                temporal_score=temporal,
                graph_score=graph,
                confidence=memory.confidence,
            )
            base_step = step_by_id.get(memory.id)
            item_trace = TraversalStep(
                from_id=base_step.from_id if base_step else f"scope:{memory.scope_id}",
                to_id=memory.id,
                relation=base_step.relation if base_step else "SEMANTIC_ANCHOR",
                depth=base_step.depth if base_step else 0,
                path=base_step.path if base_step else [f"scope:{memory.scope_id}", memory.id],
                semantic_score=semantic[memory.id],
                scope_score=retrieved.scope_score,
                temporal_score=temporal,
                graph_score=graph,
                final_score=score,
                reason=scope_access.reason,
            )
            ranked.append((retrieved, item_trace))

        ranked.sort(key=lambda pair: pair[0].score, reverse=True)
        top_items = [pair[0] for pair in ranked[:top_k]]
        packed, token_count = pack_to_token_budget(top_items, token_budget)
        selected_ids = {item.memory_id for item in packed}
        selected_trace = [step for item, step in ranked if item.memory_id in selected_ids]
        return RetrievalResult(
            items=packed,
            retrieval_latency_ms=(time.perf_counter() - started) * 1000,
            token_count=token_count,
            trace=selected_trace,
            backend_name="scopegraph",
        )

    async def _ensure_embeddings(self, memories: list[Memory]) -> None:
        missing = [
            memory
            for memory in memories
            if memory.embedding is None or memory.embedding_model != self.embedder.model_name
        ]
        if not missing:
            return
        vectors = await self.embedder.embed([memory.content for memory in missing])
        await self.repository.set_memory_embeddings(
            [(memory.id, memory.content, vector)
             for memory, vector in zip(missing, vectors, strict=True)], self.embedder.model_name,
        )
        for memory, vector in zip(missing, vectors, strict=True):
            memory.embedding = vector
            memory.embedding_model = self.embedder.model_name

    async def _eligible_memories(
        self,
        access: dict[str, ScopeAccess],
        *,
        current_scope: ScopeRef | None,
        historical: bool,
        now: datetime,
    ) -> list[Memory]:
        memories = await self.repository.list_retrieval_memories(
            scope_ids=set(access), session_id=current_scope.session_id if current_scope else None,
            historical=historical, now=now,
        )
        eligible: list[Memory] = []
        for memory in memories:
            if memory.scope_id not in access or memory.status is MemoryStatus.TOMBSTONED:
                continue
            if not historical and memory.status is not MemoryStatus.ACTIVE:
                continue
            if historical and memory.status not in {
                MemoryStatus.ACTIVE,
                MemoryStatus.SUPERSEDED,
                MemoryStatus.ARCHIVED,
            }:
                continue
            if memory.scope_level is ScopeLevel.SESSION:
                if current_scope is None or current_scope.session_id is None:
                    continue
                if memory.metadata.get("session_id") != current_scope.session_id:
                    continue
            eligible.append(memory)
        return eligible

    async def _scope_access(
        self, query: str, current_scope: ScopeRef | None
    ) -> dict[str, ScopeAccess]:
        access: dict[str, ScopeAccess] = {}
        scopes = {scope.id: scope for scope in await self.repository.list_scopes(
            include_archived=False
        )}
        if current_scope is not None:
            if current_scope.id not in scopes:
                raise ValueError("Current scope does not exist or is archived")
            access[current_scope.id] = ScopeAccess(
                self.config.current_scope_score, "same active context scope"
            )
            scope = scopes.get(current_scope.id)
            visited = {current_scope.id}
            while scope and scope.parent_scope_id and scope.parent_scope_id not in visited:
                parent = scopes.get(scope.parent_scope_id)
                if parent is None:
                    break
                visited.add(parent.id)
                access[parent.id] = ScopeAccess(
                    self.config.global_scope_score
                    if parent.scope_type is ScopeType.GLOBAL
                    else self.config.parent_scope_score,
                    "ancestor scope fallback",
                )
                scope = parent
        global_scope = next((scope for scope in scopes.values()
                             if scope.scope_type is ScopeType.GLOBAL), None)
        if global_scope is not None:
            access.setdefault(
                global_scope.id,
                ScopeAccess(self.config.global_scope_score, "global fallback scope"),
            )
        lowered = query.casefold()
        for scope in scopes.values():
            name = scope.name.casefold().strip()
            if name and name in lowered and re.search(rf"\b{re.escape(name)}\b", lowered):
                access[scope.id] = ScopeAccess(
                    max(self.config.explicit_named_scope_score,
                        access[scope.id].score if scope.id in access else 0.0),
                    f"query explicitly names scope {scope.name}",
                )
        return access

    def _memory_scope_score(
        self,
        memory: Memory,
        current_scope: ScopeRef | None,
        base_score: float,
    ) -> float:
        if (memory.scope_level is ScopeLevel.SESSION and current_scope
                and memory.metadata.get("session_id") == current_scope.session_id):
            return self.config.current_session_score
        return base_score
