from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from scopegraph.api.dependencies import get_repository
from scopegraph.graph.repository import Neo4jMemoryRepository
from scopegraph.models.scope import Scope, ScopeCreate, ScopeUpdate

router = APIRouter(prefix="/scopes", tags=["scopes"])
Repository = Annotated[Neo4jMemoryRepository, Depends(get_repository)]


@router.post("", response_model=Scope, status_code=status.HTTP_201_CREATED)
async def create_scope(request: ScopeCreate, repository: Repository) -> Scope:
    try:
        return await repository.create_scope(request)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("", response_model=list[Scope])
async def list_scopes(
    repository: Repository, include_archived: Annotated[bool, Query()] = False
) -> list[Scope]:
    return await repository.list_scopes(include_archived=include_archived)


@router.get("/{scope_id}", response_model=Scope)
async def get_scope(scope_id: str, repository: Repository) -> Scope:
    scope = await repository.get_scope(scope_id)
    if scope is None:
        raise HTTPException(status_code=404, detail="Scope not found")
    return scope


@router.patch("/{scope_id}", response_model=Scope)
async def update_scope(scope_id: str, request: ScopeUpdate, repository: Repository) -> Scope:
    scope = await repository.update_scope(scope_id, request)
    if scope is None:
        raise HTTPException(status_code=404, detail="Scope not found")
    return scope

