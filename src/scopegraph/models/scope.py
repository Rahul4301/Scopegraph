from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scopegraph.models.common import new_id, utc_now


class ScopeType(StrEnum):
    GLOBAL = "global"
    PROJECT = "project"
    REPOSITORY = "repository"
    COURSE = "course"
    CLIENT = "client"
    TASK = "task"
    WORKSPACE = "workspace"
    CUSTOM = "custom"


class ScopeRef(BaseModel):
    id: str
    name: str | None = None
    scope_type: ScopeType | None = None


class ScopeCreate(BaseModel):
    id: str = Field(default_factory=new_id)
    name: str = Field(min_length=1, max_length=200)
    scope_type: ScopeType
    parent_scope_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    archived: bool = False

    @model_validator(mode="after")
    def validate_hierarchy(self) -> "ScopeCreate":
        if self.scope_type is ScopeType.GLOBAL and self.parent_scope_id is not None:
            raise ValueError("The global root cannot have a parent scope")
        return self


class ScopeUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    parent_scope_id: str | None = None
    archived: bool | None = None


class Scope(ScopeCreate):
    model_config = ConfigDict(from_attributes=True)

