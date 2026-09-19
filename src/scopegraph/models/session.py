from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scopegraph.models.common import new_id, utc_now
from scopegraph.models.source import SourceMessageCreate


class SessionCreate(BaseModel):
    id: str = Field(default_factory=new_id)
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None
    scope_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_times(self) -> "SessionCreate":
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("ended_at cannot be before started_at")
        return self


class SessionInput(SessionCreate):
    messages: list[SourceMessageCreate] = Field(default_factory=list)


class Session(SessionCreate):
    model_config = ConfigDict(from_attributes=True)

