from datetime import UTC, datetime
from uuid import uuid4


def new_id() -> str:
    """Return a new random identifier."""
    return str(uuid4())


def utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(UTC)

