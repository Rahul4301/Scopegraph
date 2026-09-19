from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from scopegraph.api.dependencies import get_memory_system
from scopegraph.backends.scopegraph import ScopeGraphMemorySystem
from scopegraph.models.retrieval import RetrievalRequest, RetrievalResult

router = APIRouter(tags=["retrieval"])
MemoryService = Annotated[ScopeGraphMemorySystem, Depends(get_memory_system)]


@router.post("/retrieve", response_model=RetrievalResult)
async def retrieve(
    request: RetrievalRequest, memory_system: MemoryService
) -> RetrievalResult:
    try:
        return await memory_system.retrieve(
            request.query,
            current_scope=request.current_scope,
            top_k=request.top_k,
            token_budget=request.token_budget,
            now=request.now,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
