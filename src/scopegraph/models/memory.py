from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scopegraph.models.common import new_id, utc_now


class MemoryType(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    DECISION = "decision"
    EVENT = "event"
    INSTRUCTION = "instruction"
    TASK_STATE = "task_state"
    ENTITY_ATTRIBUTE = "entity_attribute"
    RELATIONSHIP = "relationship"
    SUMMARY = "summary"
    OTHER = "other"


class ScopeLevel(StrEnum):
    SESSION = "session"
    SCOPE = "scope"
    GLOBAL = "global"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    TOMBSTONED = "tombstoned"
    NEEDS_REVIEW = "needs_review"


class MemoryCreate(BaseModel):
    id: str = Field(default_factory=new_id)
    content: str = Field(min_length=1)
    memory_type: MemoryType
    scope_level: ScopeLevel
    scope_id: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    last_confirmed_at: datetime | None = None
    embedding: list[float] | None = None
    embedding_model: str | None = None
    revision: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_temporal_range(self) -> "MemoryCreate":
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from:
            raise ValueError("valid_to cannot be before valid_from")
        return self


class MemoryUpdate(BaseModel):
    content: str | None = Field(default=None, min_length=1)
    memory_type: MemoryType | None = None
    scope_level: ScopeLevel | None = None
    scope_id: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    status: MemoryStatus | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    last_confirmed_at: datetime | None = None
    embedding: list[float] | None = None
    embedding_model: str | None = None
    metadata: dict[str, Any] | None = None


class Memory(MemoryCreate):
    model_config = ConfigDict(from_attributes=True)


class MemoryCandidate(BaseModel):
    content: str = Field(min_length=1)
    memory_type: MemoryType
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    proposed_scope_level: Literal["session", "scope", "global"]
    proposed_scope_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    durability: float = Field(ge=0.0, le=1.0)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    source_message_ids: list[str] = Field(min_length=1)
    possible_contradiction: bool = False
    possible_duplicate: bool = False
    inferred: bool = False
    explicit_global_signal: bool = False

    @model_validator(mode="after")
    def validate_candidate(self) -> "MemoryCandidate":
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from:
            raise ValueError("valid_to cannot be before valid_from")
        if self.inferred and self.confidence > 0.8:
            raise ValueError("inferred memories cannot have confidence above 0.8")
        return self


class MemoryCandidateBatch(BaseModel):
    candidates: list[MemoryCandidate] = Field(default_factory=list)
