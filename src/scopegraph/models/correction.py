from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from scopegraph.models.common import new_id, utc_now
from scopegraph.models.memory import Memory


class CorrectionAction(StrEnum):
    EDIT = "edit"
    MOVE = "move"
    ARCHIVE = "archive"
    RESTORE = "restore"
    MERGE = "merge"
    TOMBSTONE = "tombstone"


class CorrectionRequest(BaseModel):
    memory_id: str
    action: CorrectionAction
    actor: str = "user"
    reason: str = ""
    changes: dict[str, Any] = Field(default_factory=dict)


class CorrectionEvent(BaseModel):
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

