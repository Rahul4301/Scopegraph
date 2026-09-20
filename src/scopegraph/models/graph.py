from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from scopegraph.models.common import utc_now
from scopegraph.models.memory import Memory
from scopegraph.models.source import SourceMessage


class GraphNode(BaseModel):
    id: str
    node_type: Literal["scope", "memory", "source_message"]
    label: str
    data: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    relation: str
    kind: str | None = None


class GraphSubgraph(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)


class MemoryProvenance(BaseModel):
    memory: Memory
    source_messages: list[SourceMessage] = Field(default_factory=list)
