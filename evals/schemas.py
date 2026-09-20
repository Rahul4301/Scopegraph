"""Typed, JSONL-safe schemas for the synthetic evaluation harness."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from scopegraph.models.scope import ScopeCreate
from scopegraph.models.session import SessionInput


class BenchmarkExample(BaseModel):
    question_id: str
    question_type: str
    question: str
    gold_answer: str
    gold_scope_ids: list[str] = Field(default_factory=list)
    gold_memory_ids: list[str] = Field(default_factory=list)
    gold_memory_contents: list[str] = Field(default_factory=list)
    gold_source_ids: list[str] = Field(default_factory=list)
    allowed_scope_ids: list[str] = Field(default_factory=list)
    current_scope_id: str | None = None
    current_session_id: str | None = None
    timestamp: datetime | None = None


class CrossScopeScenario(BaseModel):
    scenario_id: str
    difficulty: int = Field(ge=1, le=4)
    seed: int
    scopes: list[ScopeCreate]
    sessions: list[SessionInput]
    examples: list[BenchmarkExample]


class ExternalTurn(BaseModel):
    role: str
    content: str
    timestamp: datetime | None = None
    turn_id: str | None = None
    has_answer: bool = False


class ExternalSession(BaseModel):
    session_id: str
    date: datetime | None = None
    turns: list[ExternalTurn] = Field(default_factory=list)


class ExternalBenchmarkExample(BaseModel):
    dataset: str
    example_id: str
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: datetime | None = None
    sessions: list[ExternalSession] = Field(default_factory=list)
    answer_session_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdapterValidation(BaseModel):
    dataset: str
    path: str
    valid: bool
    example_count: int = 0
    session_count: int = 0
    turn_count: int = 0
    errors: list[str] = Field(default_factory=list)


class EvaluationRecord(BaseModel):
    protocol_version: str = "legacy"
    evaluation_mode: str = "unspecified"
    latency_protocol: str = "unspecified"
    embedding_preparation_ms: float | None = None
    run_id: str
    dataset: str
    system: str
    scenario_id: str
    question_id: str
    question_type: str
    question: str
    gold_answer: str
    hypothesis: str | None = None
    current_scope_id: str | None = None
    gold_scope_ids: list[str] = Field(default_factory=list)
    gold_memory_ids: list[str] = Field(default_factory=list)
    gold_source_ids: list[str] = Field(default_factory=list)
    allowed_scope_ids: list[str] = Field(default_factory=list)
    retrieved_source_ids: list[list[str]] = Field(default_factory=list)
    retrieved_origin_scope_ids: list[list[str]] = Field(default_factory=list)
    retrieved_contents: list[str] = Field(default_factory=list)
    retrieved_memory_ids: list[str] = Field(default_factory=list)
    retrieved_scope_ids: list[str] = Field(default_factory=list)
    retrieved_statuses: list[str] = Field(default_factory=list)
    retrieval_scores: list[float] = Field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    retrieved_tokens: int = 0
    answer: str | None = None
    answer_evaluated: bool = True
    answer_latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    storage_stats: dict[str, Any] = Field(default_factory=dict)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    config_hash: str
    git_commit: str | None = None
    seed: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ScoredRecord(EvaluationRecord):
    metrics: dict[str, float | None] = Field(default_factory=dict)
