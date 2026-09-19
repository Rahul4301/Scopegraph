from functools import lru_cache

from scopegraph.backends.scopegraph import ScopeGraphMemorySystem
from scopegraph.config import get_settings, load_yaml_config
from scopegraph.embeddings.cache import CachedEmbedder, SQLiteEmbeddingCache
from scopegraph.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from scopegraph.graph.client import Neo4jClient
from scopegraph.graph.repository import Neo4jMemoryRepository
from scopegraph.llm.extraction import LLMMemoryExtractor
from scopegraph.llm.openai_compatible import OpenAICompatibleLLM
from scopegraph.memory.corrections import CorrectionService
from scopegraph.memory.promoter import PromotionPolicy
from scopegraph.memory.retriever import RetrievalConfig, ScopeAwareRetriever


@lru_cache
def get_client() -> Neo4jClient:
    return Neo4jClient(get_settings())


def get_repository() -> Neo4jMemoryRepository:
    return Neo4jMemoryRepository(get_client())


def get_correction_service() -> CorrectionService:
    return CorrectionService(get_repository())


@lru_cache
def get_embedding_cache() -> SQLiteEmbeddingCache:
    return SQLiteEmbeddingCache(get_settings().embedding_cache_path)


@lru_cache
def get_memory_system() -> ScopeGraphMemorySystem:
    settings = get_settings()
    provider = OpenAICompatibleLLM(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        model=settings.llm_model,
    )
    memory_config = load_yaml_config(settings.scopegraph_config_dir / "memory.yaml")
    retrieval_config = load_yaml_config(settings.scopegraph_config_dir / "retrieval.yaml")
    embedding_provider = OpenAICompatibleEmbeddingProvider(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key.get_secret_value(),
        model=settings.embedding_model,
    )
    embedder = CachedEmbedder(embedding_provider, get_embedding_cache())
    repository = get_repository()
    return ScopeGraphMemorySystem(
        repository,
        LLMMemoryExtractor(provider),
        promotion_policy=PromotionPolicy.from_config(memory_config),
        retriever=ScopeAwareRetriever(
            repository, embedder, RetrievalConfig.from_config(retrieval_config)
        ),
    )
