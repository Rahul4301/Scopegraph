"""Measure warm local retrieval as unrelated scopes grow (no model API calls).

This microbenchmark does not measure hosted API or Neo4j latency. Use the same
command before/after changes; raw samples and environment are saved for review.
"""

import argparse
import asyncio
import json
import platform
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from scopegraph.embeddings.local import HashEmbeddingProvider
from scopegraph.graph.in_memory import InMemoryMemoryRepository
from scopegraph.memory.retriever import ScopeAwareRetriever
from scopegraph.models.memory import MemoryCreate, MemoryType, ScopeLevel
from scopegraph.models.scope import ScopeCreate, ScopeRef, ScopeType


async def measure(scope_count: int, per_scope: int, repeats: int) -> dict[str, object]:
    repo = InMemoryMemoryRepository()
    embedder = HashEmbeddingProvider()
    vector = (await embedder.embed(["project database selection"]))[0]
    for number in range(scope_count):
        scope_id = f"project-{number}"
        await repo.create_scope(ScopeCreate(id=scope_id, name=scope_id,
                                           scope_type=ScopeType.PROJECT))
        for index in range(per_scope):
            await repo.create_memory(MemoryCreate(
                id=f"{scope_id}-{index}", content=f"Database selection {index}",
                memory_type=MemoryType.FACT, scope_id=scope_id, scope_level=ScopeLevel.SCOPE,
                embedding=vector, embedding_model=embedder.model_name,
            ))
    retriever = ScopeAwareRetriever(repo, embedder)
    samples = []
    for iteration in range(repeats + 3):
        result = await retriever.retrieve("Which database?", current_scope=ScopeRef(id="project-0"),
                                          top_k=8, token_budget=1500)
        assert len(result.items) == min(8, per_scope)
        assert all(item.scope_id == "project-0" for item in result.items)
        if iteration >= 3:
            samples.append(result.retrieval_latency_ms)
    return {"memories": scope_count * per_scope, "scopes": scope_count,
            "eligible_memories": per_scope, "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "p95_ms": sorted(samples)[min(len(samples) - 1, int(len(samples) * 0.95))]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scopes", default="10,100,1000")
    parser.add_argument("--per-scope", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.per_scope < 1 or args.repeats < 1:
        parser.error("per-scope and repeats must be positive")
    rows = [asyncio.run(measure(int(count), args.per_scope, args.repeats))
            for count in args.scopes.split(",")]
    report = {"timestamp": datetime.now(UTC).isoformat(), "platform": platform.platform(),
              "python": platform.python_version(), "protocol": "warm-local-hash-v1",
              "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                       text=True, check=True).stdout.strip(),
              "measurements": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print({key: value for key, value in row.items() if key != "samples_ms"})


if __name__ == "__main__":
    main()
