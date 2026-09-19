from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from scopegraph.api.dependencies import get_client, get_embedding_cache, get_memory_system
from scopegraph.api.memories import router as memories_router
from scopegraph.api.retrieval import router as retrieval_router
from scopegraph.api.scopes import router as scopes_router
from scopegraph.api.sessions import router as sessions_router
from scopegraph.config import get_settings


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    if get_client.cache_info().currsize:
        await get_client().close()
        get_client.cache_clear()
    if get_embedding_cache.cache_info().currsize:
        get_embedding_cache().close()
        get_embedding_cache.cache_clear()
    get_memory_system.cache_clear()


app = FastAPI(title="ScopeGraph API", version="0.1.0", lifespan=lifespan)
app.include_router(scopes_router)
app.include_router(sessions_router)
app.include_router(memories_router)
app.include_router(retrieval_router)


@app.get("/health")
async def health() -> dict[str, str]:
    connected = await get_client().health()
    return {"status": "ok" if connected else "degraded", "neo4j": "up" if connected else "down"}


@app.get("/config/status")
async def config_status() -> dict[str, bool]:
    settings = get_settings()
    return {
        "llm_configured": bool(settings.llm_model and settings.llm_api_key.get_secret_value()),
        "embedding_configured": bool(
            settings.embedding_model and settings.embedding_api_key.get_secret_value()
        ),
    }
