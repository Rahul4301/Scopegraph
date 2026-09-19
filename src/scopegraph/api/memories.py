from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from scopegraph.api.dependencies import get_repository
from scopegraph.graph.repository import Neo4jMemoryRepository
from scopegraph.models.memory import Memory, MemoryCreate, MemoryUpdate

router = APIRouter(prefix="/memories", tags=["memories"])
Repository = Annotated[Neo4jMemoryRepository, Depends(get_repository)]


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


@router.patch("/{memory_id}", response_model=Memory)
async def update_memory(
    memory_id: str, request: MemoryUpdate, repository: Repository
) -> Memory:
    memory = await repository.update_memory(memory_id, request)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory

