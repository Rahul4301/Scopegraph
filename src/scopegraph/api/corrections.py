from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from scopegraph.api.dependencies import get_correction_service
from scopegraph.memory.corrections import CorrectionService
from scopegraph.models.correction import (
    CorrectionContext,
    CorrectionEvent,
    CorrectionResult,
    MemoryMergeRequest,
    MemoryMoveRequest,
    MemoryRestoreRequest,
    MemorySupersedeRequest,
    PrunePreview,
    RelationCorrectionRequest,
)

router = APIRouter(prefix="/memories", tags=["corrections"])
Corrections = Annotated[CorrectionService, Depends(get_correction_service)]


def _http_error(exc: ValueError) -> HTTPException:
    status_code = 404 if "does not exist" in str(exc) else 422
    return HTTPException(status_code=status_code, detail=str(exc))


@router.post("/{memory_id}/move", response_model=CorrectionResult)
async def move_memory(
    memory_id: str, request: MemoryMoveRequest, corrections: Corrections
) -> CorrectionResult:
    try:
        return await corrections.move(memory_id, request)
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.post("/{memory_id}/archive", response_model=CorrectionResult)
async def archive_memory(
    memory_id: str, request: CorrectionContext, corrections: Corrections
) -> CorrectionResult:
    try:
        return await corrections.archive(
            memory_id, actor=request.actor, reason=request.reason
        )
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.post("/{memory_id}/prune/preview", response_model=PrunePreview)
async def preview_prune(memory_id: str, corrections: Corrections) -> PrunePreview:
    try:
        return await corrections.preview_prune(memory_id)
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.post("/{memory_id}/prune", response_model=CorrectionResult)
async def prune_memory(
    memory_id: str, request: CorrectionContext, corrections: Corrections
) -> CorrectionResult:
    try:
        return await corrections.prune(
            memory_id, actor=request.actor, reason=request.reason
        )
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.post("/{memory_id}/restore", response_model=CorrectionResult)
async def restore_memory(
    memory_id: str, request: MemoryRestoreRequest, corrections: Corrections
) -> CorrectionResult:
    try:
        return await corrections.restore(memory_id, request)
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.post("/{memory_id}/merge", response_model=CorrectionResult)
async def merge_memory(
    memory_id: str, request: MemoryMergeRequest, corrections: Corrections
) -> CorrectionResult:
    try:
        return await corrections.merge(memory_id, request)
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.post("/{memory_id}/supersede", response_model=CorrectionResult)
async def supersede_memory(
    memory_id: str, request: MemorySupersedeRequest, corrections: Corrections
) -> CorrectionResult:
    try:
        return await corrections.supersede(memory_id, request)
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.post("/{memory_id}/relations/add", response_model=CorrectionResult)
async def add_memory_relation(
    memory_id: str, request: RelationCorrectionRequest, corrections: Corrections
) -> CorrectionResult:
    try:
        return await corrections.add_relation(memory_id, request)
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.post("/{memory_id}/relations/remove", response_model=CorrectionResult)
async def remove_memory_relation(
    memory_id: str, request: RelationCorrectionRequest, corrections: Corrections
) -> CorrectionResult:
    try:
        return await corrections.remove_relation(memory_id, request)
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.get("/{memory_id}/history", response_model=list[CorrectionEvent])
async def memory_history(
    memory_id: str, corrections: Corrections
) -> list[CorrectionEvent]:
    try:
        return await corrections.history(memory_id)
    except ValueError as exc:
        raise _http_error(exc) from exc
