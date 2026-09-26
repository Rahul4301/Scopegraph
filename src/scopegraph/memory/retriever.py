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
from scopegraph.models.source import SourceMessage
from scopegraph.observability.token_counting import pack_to_token_budget

WORD_PATTERN = re.compile(r"[\w']+", re.UNICODE)
STOP_WORDS = {
    "a", "an", "and", "are", "did", "do", "does", "for", "from", "has", "have",
    "how", "in", "is", "it", "of", "on", "the", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "would", "could", "likely",
}


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

    async def get_source_messages_by_ids(
        self, message_ids: list[str]
    ) -> list[SourceMessage]: ...

    async def list_source_messages_for_scopes(
        self, scope_ids: set[str], *, now: datetime, session_id: str | None
    ) -> list[tuple[SourceMessage, str]]: ...


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
    scope_access_mode: str = "hierarchical"
    enforce_temporal_status: bool = True
    enforce_session_access: bool = True
    candidate_pool_size: int = 64
    raw_source_candidates: int = 0
    lexical_weight: float = 0.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RetrievalConfig":
        scope = config.get("scope", {})
        result = cls(
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
            scope_access_mode=str(config.get("scope_access_mode", "hierarchical")),
            enforce_temporal_status=bool(config.get("enforce_temporal_status", True)),
            enforce_session_access=bool(config.get("enforce_session_access", True)),
            candidate_pool_size=int(config.get("candidate_pool_size", 64)),
            raw_source_candidates=int(config.get("raw_source_candidates", 0)),
            lexical_weight=float(config.get("lexical_weight", 0.0)),
        )
        if result.scope_access_mode not in {"hierarchical", "all"}:
            raise ValueError("scope_access_mode must be 'hierarchical' or 'all'")
        return result


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
        historical = (
            self.config.enforce_temporal_status
            and is_historical_query(query, self.config.historical_query_terms)
        )
        candidates = await self._eligible_memories(
            access, current_scope=current_scope, historical=historical, now=effective_now
        )
        if not candidates and self.config.raw_source_candidates <= 0:
            return RetrievalResult(
                items=[], trace=[], token_count=0, backend_name="scopegraph",
                retrieval_latency_ms=(time.perf_counter() - started) * 1000,
            )
        query_vector = (await self.embedder.embed([query]))[0]
        await self._ensure_embeddings(candidates)
        semantic = {
            memory.id: cosine_similarity(query_vector, memory.embedding or [])
            for memory in candidates
        }
        traversal_enabled = self.config.max_graph_hops > 0 and self.config.max_expanded_nodes > 0
        pool_size = min(len(candidates), max(top_k, self.config.candidate_pool_size))
        anchor_limit = min(
            len(candidates),
            max(1, top_k // 2) if traversal_enabled else pool_size,
        )
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
        # Semantic candidates always remain in the final reranking pool. Graph
        # expansion augments this pool rather than displacing stronger matches.
        semantic_pool = nlargest(pool_size, candidates, key=lambda memory: semantic[memory.id])
        all_memories = {memory.id: memory for memory in semantic_pool}
        all_memories.update({memory.id: memory for memory in expanded_memories})
        anchor_ids = {memory.id for memory in anchors}
        step_by_id = {step.to_id: step for step in traversal_steps}
        ranked: list[tuple[RetrievedMemory, TraversalStep]] = []
        for memory in all_memories.values():
            temporal = (
                temporal_score(memory, now=effective_now, historical=historical)
                if self.config.enforce_temporal_status
                else 1.0
            )
            if temporal == 0:
                continue
            depth = expanded.get(memory.id, (memory, 0))[1]
            graph = 0.0 if memory.id in anchor_ids or depth == 0 else 1.0 / depth
            scope_access = access[memory.scope_id]
            recency = recency_score(memory, now=effective_now)
            base_score = final_score(
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
            lexical = self._lexical_score(query, memory.content)
            score = (
                (1.0 - self.config.lexical_weight) * base_score
                + self.config.lexical_weight * lexical
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

        # Search original turns independently so extraction omissions and lossy
        # summaries cannot make source evidence unreachable.
        source_rows = (
            await self.repository.list_source_messages_for_scopes(
                set(access), now=effective_now,
                session_id=current_scope.session_id if current_scope else None,
            )
            if self.config.raw_source_candidates > 0
            else []
        )
        if source_rows:
            source_vectors = await self.embedder.embed([row[0].content for row in source_rows])
            raw_ranked: list[tuple[float, SourceMessage, str, float]] = []
            for (source, scope_id), vector in zip(source_rows, source_vectors, strict=True):
                source_semantic = cosine_similarity(query_vector, vector)
                lexical = self._lexical_score(query, source.content)
                base_score = final_score(
                    semantic=source_semantic,
                    scope=access[scope_id].score,
                    temporal=1.0,
                    confidence=1.0,
                    graph=0.0,
                    recency=1.0,
                    weights=self.config.weights,
                )
                score = (
                    (1.0 - self.config.lexical_weight) * base_score
                    + self.config.lexical_weight * lexical
                )
                raw_ranked.append((score, source, scope_id, source_semantic))
            raw_ranked.sort(key=lambda row: row[0], reverse=True)
            by_session = {
                source.session_id: sorted(
                    [item for item, _ in source_rows if item.session_id == source.session_id],
                    key=lambda item: item.turn_index,
                )
                for source, _ in source_rows
            }
            for score, source, scope_id, source_semantic in raw_ranked[
                : self.config.raw_source_candidates
            ]:
                neighbors = [
                    item for item in by_session[source.session_id]
                    if abs(item.turn_index - source.turn_index) <= 1
                ]
                item = RetrievedMemory(
                    memory_id=f"source:{source.id}",
                    content="Original conversation evidence",
                    score=score,
                    scope_id=scope_id,
                    scope_level=ScopeLevel.SCOPE,
                    status=MemoryStatus.ACTIVE,
                    source_ids=[source.id],
                    source_messages=neighbors,
                    valid_from=source.timestamp,
                    semantic_score=source_semantic,
                    scope_score=access[scope_id].score,
                    temporal_score=1.0,
                    confidence=1.0,
                )
                ranked.append((item, TraversalStep(
                    from_id=f"scope:{scope_id}", to_id=item.memory_id,
                    relation="RAW_SOURCE", depth=0,
                    path=[f"scope:{scope_id}", item.memory_id],
                    semantic_score=source_semantic, scope_score=access[scope_id].score,
                    temporal_score=1.0, graph_score=0.0, final_score=score,
                    reason="original source turn",
                )))

        ranked.sort(key=lambda pair: pair[0].score, reverse=True)
        top_items: list[RetrievedMemory] = []
        covered_source_ids: set[str] = set()
        for item, _ in ranked:
            if item.source_ids and set(item.source_ids) <= covered_source_ids:
                continue
            top_items.append(item)
            covered_source_ids.update(item.source_ids)
            if len(top_items) >= top_k:
                break
        source_ids = list(dict.fromkeys(
            source_id
            for item in top_items
            for source_id in item.source_ids
        ))
        sources = await self.repository.get_source_messages_by_ids(source_ids)
        sources_by_id = {source.id: source for source in sources}
        with_provenance = [
            item.model_copy(update={
                "source_messages": item.source_messages or [
                    sources_by_id[source_id] for source_id in item.source_ids
                    if source_id in sources_by_id]
            })
            for item in top_items
        ]
        packed, token_count = pack_to_token_budget(with_provenance, token_budget, query=query)
        selected_ids = {item.memory_id for item in packed}
        selected_trace = [step for item, step in ranked if item.memory_id in selected_ids]
        return RetrievalResult(
            items=packed,
            retrieval_latency_ms=(time.perf_counter() - started) * 1000,
            token_count=token_count,
            trace=selected_trace,
            backend_name="scopegraph",
        )

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        query_terms = {
            term for term in WORD_PATTERN.findall(query.casefold())
            if len(term) > 1 and term not in STOP_WORDS
        }
        if not query_terms:
            return 0.0
        text_terms = set(WORD_PATTERN.findall(text.casefold()))
        return len(query_terms & text_terms) / len(query_terms)

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
        if self.config.enforce_temporal_status:
            memories = await self.repository.list_retrieval_memories(
                scope_ids=set(access),
                session_id=current_scope.session_id if current_scope else None,
                historical=historical,
                now=now,
            )
        else:
            # Deliberately expose inactive and out-of-date candidates for the
            # temporal/status ablation. Tombstones remain excluded because they
            # represent deletion, not temporal ranking behavior.
            memories = await self.repository.list_memories(include_inactive=True)
        eligible: list[Memory] = []
        for memory in memories:
            if memory.scope_id not in access or memory.status is MemoryStatus.TOMBSTONED:
                continue
            if self.config.enforce_temporal_status:
                if not historical and memory.status is not MemoryStatus.ACTIVE:
                    continue
                if historical and memory.status not in {
                    MemoryStatus.ACTIVE,
                    MemoryStatus.SUPERSEDED,
                    MemoryStatus.ARCHIVED,
                }:
                    continue
            if self.config.enforce_session_access and memory.scope_level is ScopeLevel.SESSION:
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
        if self.config.scope_access_mode == "all":
            return {
                scope.id: ScopeAccess(self.config.current_scope_score, "scope hierarchy disabled")
                for scope in scopes.values()
            }
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
