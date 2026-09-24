"""Provider construction for reproducible offline and live evaluations."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from evals.adapters.cross_scope_mem import KeywordEmbeddingProvider, ScenarioExtractor
from evals.scenarios.generate_cross_scope import candidates_by_message
from evals.schemas import CrossScopeScenario
from scopegraph.config import get_settings
from scopegraph.embeddings.base import EmbeddingProvider
from scopegraph.embeddings.cache import CachedEmbedder, SQLiteEmbeddingCache
from scopegraph.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from scopegraph.llm.extraction import CandidateExtractor, LLMMemoryExtractor
from scopegraph.llm.openai_compatible import OpenAICompatibleLLM
from scopegraph.llm.transport import close_provider
from scopegraph.models.memory import MemoryCandidate
from scopegraph.models.scope import ScopeRef
from scopegraph.models.source import SourceMessage


class PreparedEmbedder:
    """In-process cache: timed retrieval never pays another system's API warmup."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider
        self.model_name = provider.model_name
        self.vectors: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with self._lock:
            missing = list(dict.fromkeys(text for text in texts if text not in self.vectors))
            if missing:
                vectors = await self.provider.embed(missing)
                self.vectors.update(zip(missing, vectors, strict=True))
            return [list(self.vectors[text]) for text in texts]


async def freeze_extraction(
    scenario: CrossScopeScenario,
    providers: "EvaluationProviders",
    *,
    cached: dict[str, list[MemoryCandidate]] | None = None,
    checkpoint: Callable[[dict[str, list[MemoryCandidate]]], None] | None = None,
) -> dict[str, list[MemoryCandidate]]:
    """Extract each source once, independently of mutable repository state."""
    by_message: dict[str, list[MemoryCandidate]] = dict(cached or {})
    scopes = {scope.id: scope for scope in scenario.scopes}
    for session in scenario.sessions:
        if not session.messages or all(message.id in by_message for message in session.messages):
            continue
        scope = scopes[session.scope_id]
        candidates = await providers.extractor.extract(
            [SourceMessage(**message.model_dump()) for message in session.messages],
            current_scope=ScopeRef(
                id=scope.id, name=scope.name, scope_type=scope.scope_type, session_id=session.id
            ),
            existing_memories=[],
        )
        message_ids = {message.id for message in session.messages}
        invalid_sources = sorted(
            {
                source_id
                for candidate in candidates
                for source_id in candidate.source_message_ids
                if source_id not in message_ids
            }
        )
        if invalid_sources:
            raise ValueError(
                f"Extractor returned source IDs outside session {session.id}: "
                f"{invalid_sources}"
            )
        for message in session.messages:
            by_message[message.id] = [
                candidate
                for candidate in candidates
                if message.id in candidate.source_message_ids
            ]
        if checkpoint is not None:
            checkpoint(by_message)
    await close_provider(providers.extractor)
    providers.extractor = ScenarioExtractor(by_message)
    return by_message


@dataclass
class EvaluationProviders:
    extractor: CandidateExtractor
    embedder: EmbeddingProvider
    embedding_cache: SQLiteEmbeddingCache | None = None

    def close(self) -> None:
        if self.embedding_cache is not None:
            self.embedding_cache.close()

    async def aclose(self) -> None:
        await close_provider(self.extractor)
        await close_provider(self.embedder)
        self.close()


def build_cross_scope_providers(
    scenario: CrossScopeScenario,
    *,
    live_extraction: bool,
    live_embeddings: bool,
) -> EvaluationProviders:
    settings = get_settings()
    if live_extraction:
        extractor: CandidateExtractor = LLMMemoryExtractor(
            OpenAICompatibleLLM(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key.get_secret_value(),
                model=settings.llm_model,
            )
        )
    else:
        extractor = ScenarioExtractor(candidates_by_message(scenario))

    if live_embeddings:
        cache = SQLiteEmbeddingCache(settings.embedding_cache_path)
        embedder: EmbeddingProvider = CachedEmbedder(
            OpenAICompatibleEmbeddingProvider(
                base_url=settings.embedding_base_url,
                api_key=settings.embedding_api_key.get_secret_value(),
                model=settings.embedding_model,
            ),
            cache,
        )
        return EvaluationProviders(extractor, PreparedEmbedder(embedder), cache)
    return EvaluationProviders(extractor, PreparedEmbedder(KeywordEmbeddingProvider()))
