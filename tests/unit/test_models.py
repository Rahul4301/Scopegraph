from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from scopegraph.models.memory import MemoryCreate, MemoryType, ScopeLevel
from scopegraph.models.scope import ScopeCreate, ScopeType
from scopegraph.models.session import SessionCreate


def test_global_scope_rejects_parent() -> None:
    with pytest.raises(ValidationError, match="global root"):
        ScopeCreate(name="global", scope_type=ScopeType.GLOBAL, parent_scope_id="parent")


def test_session_rejects_backwards_time_range() -> None:
    started = datetime.now(UTC)
    with pytest.raises(ValidationError, match="ended_at"):
        SessionCreate(
            scope_id="scope", started_at=started, ended_at=started - timedelta(seconds=1)
        )


def test_memory_rejects_invalid_confidence_and_validity() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError):
        MemoryCreate(
            content="Alpha uses Neo4j",
            memory_type=MemoryType.FACT,
            scope_level=ScopeLevel.SCOPE,
            scope_id="alpha",
            confidence=1.1,
            valid_from=now,
            valid_to=now - timedelta(days=1),
        )

