from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scopegraph.models.common import new_id, utc_now
from scopegraph.models.memory import Memory, MemoryStatus, MemoryType, ScopeLevel
from scopegraph.models.relationship import RelationKind


class CorrectionAction(StrEnum):
    EDIT = "edit"
    MOVE_SCOPE = "move_scope"
    ARCHIVE = "archive"
    RESTORE = "restore"
    MERGE = "merge"
    TOMBSTONE = "tombstone"
    SUPERSEDE = "supersede"
    REMOVE_RELATION = "remove_relation"
    ADD_RELATION = "add_relation"


class CorrectionRelation(StrEnum):
    SUPPORTS = "SUPPORTS"
    SAME_AS = "SAME_AS"
    CONTRADICTS = "CONTRADICTS"
    RELATES_TO = "RELATES_TO"


class CorrectionRequest(BaseModel):
    memory_id: str
    action: CorrectionAction
    actor: str = "user"
    reason: str = ""
    changes: dict[str, Any] = Field(default_factory=dict)
    undo_of: str | None = None


class CorrectionEvent(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str = Field(default_factory=new_id)
    action: CorrectionAction
    timestamp: datetime = Field(default_factory=utc_now)
    actor: str
    before: dict[str, Any]
    after: dict[str, Any]
    reason: str = ""
    undo_of: str | None = None


class CorrectionResult(BaseModel):
    event: CorrectionEvent
    memory: Memory
    affected_memories: list[Memory] = Field(default_factory=list)


class CorrectionContext(BaseModel):
    actor: str = "user"
    reason: str = ""


class MemoryEditRequest(CorrectionContext):
    content: str | None = Field(default=None, min_length=1)
    memory_type: MemoryType | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    metadata: dict[str, Any] | None = None


class MemoryMoveRequest(CorrectionContext):
    scope_id: str
    scope_level: ScopeLevel | None = None


class MemoryMergeRequest(CorrectionContext):
    target_memory_id: str


class MemoryRestoreRequest(CorrectionContext):
    undo_of: str | None = None


class MemorySupersedeRequest(CorrectionContext):
    replacement_memory_id: str


class RelationCorrectionRequest(CorrectionContext):
    target_memory_id: str
    relation: CorrectionRelation
    kind: RelationKind | None = None

    @model_validator(mode="after")
    def validate_relation_kind(self) -> "RelationCorrectionRequest":
        if self.relation is CorrectionRelation.RELATES_TO and self.kind is None:
            raise ValueError("RELATES_TO requires an allowlisted kind")
        if self.relation is not CorrectionRelation.RELATES_TO and self.kind is not None:
            raise ValueError("kind is only valid for RELATES_TO")
        return self


class SupportDependency(BaseModel):
    memory: Memory
    other_active_support_ids: list[str] = Field(default_factory=list)


class PruneImpact(BaseModel):
    memory_id: str
    relation: str = "SUPPORTS"
    current_status: MemoryStatus
    proposed_status: MemoryStatus
    other_active_support_ids: list[str] = Field(default_factory=list)
    reason: str


class GraphNeighborPreview(BaseModel):
    memory_id: str
    relation: str
    evidentiary_dependency: bool = False


class PrunePreview(BaseModel):
    memory_id: str
    current_status: MemoryStatus
    proposed_status: Literal[MemoryStatus.TOMBSTONED] = MemoryStatus.TOMBSTONED
    dependencies: list[PruneImpact] = Field(default_factory=list)
    graph_neighbors: list[GraphNeighborPreview] = Field(default_factory=list)
    hard_delete: bool = False
