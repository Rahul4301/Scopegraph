"""Provider construction for reproducible offline and live evaluations."""

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
from scopegraph.models.memory import MemoryCandidate
from scopegraph.models.scope import ScopeRef
from scopegraph.models.source import SourceMessage


class PreparedEmbedder:
    """In-process cache: timed retrieval never pays another system's API warmup."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider
        self.model_name = provider.model_name
        self.vectors: dict[str, list[float]] = {}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(text for text in texts if text not in self.vectors))
        if missing:
            vectors = await self.provider.embed(missing)
            self.vectors.update(zip(missing, vectors, strict=True))
        return [list(self.vectors[text]) for text in texts]


async def freeze_extraction(scenario: CrossScopeScenario,
                            providers: "EvaluationProviders") -> dict[str, list[MemoryCandidate]]:
    """Extract each source once, with identical inputs independent of backend state."""
    by_message: dict[str, list[MemoryCandidate]] = {}
    scopes = {scope.id: scope for scope in scenario.scopes}
    for session in scenario.sessions:
        scope = scopes[session.scope_id]
        candidates = await providers.extractor.extract(
            [SourceMessage(**message.model_dump()) for message in session.messages],
            current_scope=ScopeRef(id=scope.id, name=scope.name, scope_type=scope.scope_type,
                                   session_id=session.id),
            existing_memories=[],
        )
        # ScenarioExtractor consumes each candidate exactly once even if it cites
        # several messages from the same session.
        by_message[session.messages[0].id] = candidates
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
        extractor = ScenarioExtractor(
            candidates_by_message(scenario)
        )

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
