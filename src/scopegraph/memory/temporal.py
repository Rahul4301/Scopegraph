from datetime import datetime

from scopegraph.models.memory import Memory


def supersession_time(new_memory: Memory) -> datetime:
    return new_memory.valid_from or new_memory.created_at
