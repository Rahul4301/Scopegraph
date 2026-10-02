"""Common schema shared by every loader, arm, and runner."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Turn(BaseModel):
    speaker: str
    text: str


class Session(BaseModel):
    session_id: str
    timestamp: str | None = None
    turns: list[Turn]


class History(BaseModel):
    """One stable history. Every history maps to exactly one Mem0 user_id."""

    history_id: str
    bench: str
    sessions: list[Session] = Field(default_factory=list)
    document: str | None = None  # verbatim text for benchmarks that are not session-structured
    # As-of querying: a checkpoint history is a prefix of a conversation and shares that
    # conversation's memory store, so its store_id is the full conversation's history_id.
    store_id: str | None = None

    @property
    def user_id(self) -> str:
        return make_user_id(self.bench, self.history_id)

    @property
    def store_key(self) -> str:
        """Scope of the memory store a history reads and writes. Equal to user_id except for
        as-of checkpoints, which all share the store of the conversation they are cut from."""
        return make_user_id(self.bench, self.store_id or self.history_id)


class Item(BaseModel):
    """One scored question (or one coding task) tied to a history."""

    item_id: str
    bench: str
    history_id: str
    category: str
    question: str
    gold: str
    question_date: str | None = None
    kind: Literal["chat", "coding"] = "chat"
    primary: bool = True
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _safe_id(self) -> Item:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.item_id):
            raise ValueError(f"item_id must be filename safe: {self.item_id!r}")
        return self


class Usage(BaseModel):
    """Token usage of one model call.

    output_tokens already includes reasoning_tokens (OpenAI reports reasoning as a subset of
    output), so cost never adds the two together.
    """

    input_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @model_validator(mode="after")
    def _consistent(self) -> Usage:
        if self.cached_tokens + self.cache_write_tokens > self.input_tokens:
            raise ValueError("cached + cache_write tokens exceed input tokens")
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning tokens exceed output tokens; output must include them")
        return self

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


def make_user_id(bench: str, history_id: str) -> str:
    """user_id = "<bench>_<history_id>". Never shared across test cases."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", history_id)
    return f"{bench}_{safe}"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value)


def render_session(session: Session) -> str:
    header = f"[Session {session.session_id}"
    header += f" | {session.timestamp}]" if session.timestamp else "]"
    lines = [header] + [f"{t.speaker}: {t.text}" for t in session.turns]
    return "\n".join(lines)


def render_transcript(history: History) -> str:
    """Deterministic transcript. Arm A sends this verbatim, arm C chunks it. A history that is
    one verbatim document is returned unchanged."""
    if history.document is not None:
        return history.document
    return "\n\n".join(render_session(s) for s in history.sessions)
