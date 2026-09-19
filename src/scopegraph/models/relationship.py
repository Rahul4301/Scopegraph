from enum import StrEnum

from pydantic import BaseModel


class RelationKind(StrEnum):
    USES = "USES"
    PREFERS = "PREFERS"
    WORKS_ON = "WORKS_ON"
    DEPENDS_ON = "DEPENDS_ON"
    DECIDED = "DECIDED"
    REPLACED = "REPLACED"
    MENTIONED_WITH = "MENTIONED_WITH"
    MEMBER_OF = "MEMBER_OF"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"


class MemoryRelationship(BaseModel):
    source_memory_id: str
    target_memory_id: str
    kind: RelationKind

