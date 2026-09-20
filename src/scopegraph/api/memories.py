from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from scopegraph.api.dependencies import get_correction_service, get_repository
from scopegraph.graph.repository import Neo4jMemoryRepository
from scopegraph.memory.corrections import CorrectionService
from scopegraph.models.correction import CorrectionResult, MemoryEditRequest
from scopegraph.models.graph import MemoryProvenance
from scopegraph.models.memory import Memory, MemoryCreate

router = APIRouter(prefix="/memories", tags=["memories"])
Repository = Annotated[Neo4jMemoryRepository, Depends(get_repository)]
Corrections = Annotated[CorrectionService, Depends(get_correction_service)]


@router.post("", response_model=Memory, status_code=status.HTTP_201_CREATED)
async def create_memory(request: MemoryCreate, repository: Repository) -> Memory:
    try:
        return await repository.create_memory(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("", response_model=list[Memory])
async def list_memories(
    repository: Repository,
    scope_id: Annotated[str | None, Query()] = None,
    include_inactive: Annotated[bool, Query()] = False,
) -> list[Memory]:
    return await repository.list_memories(
        scope_id=scope_id, include_inactive=include_inactive
    )


@router.get("/{memory_id}", response_model=Memory)
async def get_memory(memory_id: str, repository: Repository) -> Memory:
    memory = await repository.get_memory(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


@router.get("/{memory_id}/provenance", response_model=MemoryProvenance)
async def get_memory_provenance(
    memory_id: str, repository: Repository
) -> MemoryProvenance:
    memory = await repository.get_memory(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    messages = await repository.get_source_messages_by_ids(memory.source_ids)
    return MemoryProvenance(memory=memory, source_messages=messages)


@router.patch("/{memory_id}", response_model=CorrectionResult)
async def update_memory(
    memory_id: str, request: MemoryEditRequest, corrections: Corrections
) -> CorrectionResult:
    try:
        return await corrections.edit(memory_id, request)
    except ValueError as exc:
        code = 404 if "does not exist" in str(exc) else 422
        raise HTTPException(status_code=code, detail=str(exc)) from exc
