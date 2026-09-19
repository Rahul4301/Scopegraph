from functools import lru_cache

from scopegraph.backends.scopegraph import ScopeGraphMemorySystem
from scopegraph.config import get_settings, load_yaml_config
from scopegraph.graph.client import Neo4jClient
from scopegraph.graph.repository import Neo4jMemoryRepository
from scopegraph.llm.extraction import LLMMemoryExtractor
from scopegraph.llm.openai_compatible import OpenAICompatibleLLM
from scopegraph.memory.promoter import PromotionPolicy


@lru_cache
def get_client() -> Neo4jClient:
    return Neo4jClient(get_settings())


def get_repository() -> Neo4jMemoryRepository:
    return Neo4jMemoryRepository(get_client())


def get_memory_system() -> ScopeGraphMemorySystem:
    settings = get_settings()
    provider = OpenAICompatibleLLM(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        model=settings.llm_model,
    )
    memory_config = load_yaml_config(settings.scopegraph_config_dir / "memory.yaml")
    return ScopeGraphMemorySystem(
        get_repository(),
        LLMMemoryExtractor(provider),
        promotion_policy=PromotionPolicy.from_config(memory_config),
    )
