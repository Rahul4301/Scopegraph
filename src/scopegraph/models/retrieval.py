from datetime import datetime

from pydantic import BaseModel, Field

from scopegraph.models.memory import MemoryStatus, ScopeLevel


class TraversalStep(BaseModel):
    from_id: str
    to_id: str
    relation: str
    depth: int = Field(ge=0)


class RetrievedMemory(BaseModel):
    memory_id: str
    content: str
    score: float
    scope_id: str
    scope_level: ScopeLevel
    status: MemoryStatus
    source_ids: list[str] = Field(default_factory=list)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    semantic_score: float = 0.0
    scope_score: float = 0.0
    temporal_score: float = 0.0
    graph_score: float = 0.0
    confidence: float = Field(ge=0.0, le=1.0)


class RetrievalResult(BaseModel):
    items: list[RetrievedMemory]
    retrieval_latency_ms: float = Field(ge=0.0)
    token_count: int = Field(ge=0)
    trace: list[TraversalStep]
    backend_name: str


class IngestResult(BaseModel):
    session_id: str
    source_message_ids: list[str] = Field(default_factory=list)
    memory_ids: list[str] = Field(default_factory=list)
    duplicate_count: int = 0
    conflict_count: int = 0
    promoted_count: int = 0


class MemoryStats(BaseModel):
    backend_name: str
    scope_count: int = 0
    session_count: int = 0
    source_message_count: int = 0
    memory_count: int = 0
    relationship_count: int = 0
