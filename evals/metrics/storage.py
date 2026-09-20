"""Logical storage metrics independent of Neo4j file layout."""

from typing import Any


def logical_storage_stats(stats: Any, *, serialized_memory_bytes: int = 0,
                          embedding_count: int = 0) -> dict[str, int | float]:
    values = stats.model_dump() if hasattr(stats, "model_dump") else dict(stats)
    nodes = sum(int(values.get(key, 0)) for key in
                ("scope_count", "session_count", "source_message_count", "memory_count"))
    edges = int(values.get("relationship_count", 0))
    return {
        "nodes": nodes, "edges": edges,
        "memory_count": int(values.get("memory_count", 0)),
        "embedding_count": int(embedding_count),
        "logical_bytes": int(serialized_memory_bytes),
    }
