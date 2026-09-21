import asyncio
import time
from dataclasses import dataclass
from typing import Protocol

from scopegraph.models.memory import Memory
from scopegraph.models.retrieval import TraversalStep


@dataclass(frozen=True)
class MemoryNeighbor:
    source_id: str
    memory: Memory
    relation: str


class TraversalRepository(Protocol):
    async def get_memory_neighbors(
        self, memory_ids: list[str], *, eligible_ids: set[str] | None = None,
        limit: int | None = None,
    ) -> list[MemoryNeighbor]: ...


async def bounded_traversal(
    repository: TraversalRepository,
    anchor_ids: list[str],
    *,
    allowed_scope_ids: set[str],
    max_hops: int,
    max_expanded_nodes: int,
    max_time_ms: float = 100.0,
    eligible_ids: set[str] | None = None,
) -> tuple[dict[str, tuple[Memory, int]], list[TraversalStep]]:
    started = time.perf_counter()
    visited = set(anchor_ids)
    frontier = list(anchor_ids)
    paths = {anchor_id: [anchor_id] for anchor_id in anchor_ids}
    expanded: dict[str, tuple[Memory, int]] = {}
    steps: list[TraversalStep] = []
    for depth in range(1, max_hops + 1):
        if not frontier or len(expanded) >= max_expanded_nodes:
            break
        if (time.perf_counter() - started) * 1000 >= max_time_ms:
            break
        remaining_seconds = max_time_ms / 1000 - (time.perf_counter() - started)
        try:
            async with asyncio.timeout(remaining_seconds):
                neighbors = await repository.get_memory_neighbors(
                    frontier, eligible_ids=eligible_ids,
                    limit=max_expanded_nodes + len(visited),
                )
        except TimeoutError:
            break
        next_frontier: list[str] = []
        for neighbor in neighbors:
            memory = neighbor.memory
            if memory.id in visited or memory.scope_id not in allowed_scope_ids:
                continue
            if eligible_ids is not None and memory.id not in eligible_ids:
                continue
            visited.add(memory.id)
            expanded[memory.id] = (memory, depth)
            next_frontier.append(memory.id)
            path = [*paths[neighbor.source_id], memory.id]
            paths[memory.id] = path
            steps.append(
                TraversalStep(
                    from_id=neighbor.source_id,
                    to_id=memory.id,
                    relation=neighbor.relation,
                    depth=depth,
                    path=path,
                )
            )
            if len(expanded) >= max_expanded_nodes:
                break
        frontier = next_frontier
    return expanded, steps
