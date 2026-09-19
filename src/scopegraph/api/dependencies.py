from functools import lru_cache

from scopegraph.config import get_settings
from scopegraph.graph.client import Neo4jClient
from scopegraph.graph.repository import Neo4jMemoryRepository


@lru_cache
def get_client() -> Neo4jClient:
    return Neo4jClient(get_settings())


def get_repository() -> Neo4jMemoryRepository:
    return Neo4jMemoryRepository(get_client())

