from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from scopegraph.api.dependencies import get_memory_system, get_repository
from scopegraph.backends.scopegraph import ScopeGraphMemorySystem
from scopegraph.graph.repository import Neo4jMemoryRepository
from scopegraph.models.retrieval import IngestResult
from scopegraph.models.session import Session, SessionConsolidateRequest, SessionCreate
from scopegraph.models.source import SourceMessage, SourceMessageCreate

router = APIRouter(prefix="/sessions", tags=["sessions"])
Repository = Annotated[Neo4jMemoryRepository, Depends(get_repository)]
MemoryService = Annotated[ScopeGraphMemorySystem, Depends(get_memory_system)]


@router.post("", response_model=Session, status_code=status.HTTP_201_CREATED)
async def create_session(request: SessionCreate, repository: Repository) -> Session:
    try:
        return await repository.create_session(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{session_id}", response_model=Session)
async def get_session(session_id: str, repository: Repository) -> Session:
    session = await repository.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/{session_id}/messages", response_model=SourceMessage, status_code=201)
async def create_message(
    session_id: str, request: SourceMessageCreate, repository: Repository
) -> SourceMessage:
    if request.session_id != session_id:
        raise HTTPException(status_code=422, detail="Path and body session IDs differ")
    try:
        return await repository.create_source_message(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/{session_id}/consolidate", response_model=IngestResult)
async def consolidate_session(
    session_id: str,
    request: SessionConsolidateRequest,
    memory_system: MemoryService,
) -> IngestResult:
    try:
        return await memory_system.consolidate_session(
            session_id, current_scope=request.current_scope
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
