from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from scopegraph.models.common import new_id, utc_now


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class SourceMessageCreate(BaseModel):
    id: str = Field(default_factory=new_id)
    session_id: str
    role: MessageRole
    content: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=utc_now)
    turn_index: int = Field(ge=0)


class SourceMessage(SourceMessageCreate):
    model_config = ConfigDict(from_attributes=True)

