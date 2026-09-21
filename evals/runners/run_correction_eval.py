"""Run the three-condition correction-persistence experiment."""

import argparse
import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from evals.adapters.cross_scope_mem import KeywordEmbeddingProvider, ScenarioExtractor
from evals.metrics.correction_persistence import error_relapse_rate
from scopegraph.backends.scopegraph import ScopeGraphMemorySystem
from scopegraph.graph.in_memory import InMemoryMemoryRepository
from scopegraph.memory.retriever import ScopeAwareRetriever
from scopegraph.models.correction import MemoryEditRequest
from scopegraph.models.memory import MemoryCandidate, MemoryType
from scopegraph.models.scope import ScopeCreate, ScopeRef, ScopeType
from scopegraph.models.session import SessionInput
from scopegraph.models.source import MessageRole, SourceMessageCreate


async def _condition(condition: str, probes: tuple[int, ...]) -> dict[str, object]:
    repository = InMemoryMemoryRepository()
    await repository.create_scope(ScopeCreate(
        id="global", name="Global", scope_type=ScopeType.GLOBAL,
        created_at=datetime(2026, 6, 10, tzinfo=UTC),
    ))
    await repository.create_scope(ScopeCreate(
        id="scope_alpha", name="Alpha", scope_type=ScopeType.PROJECT,
        parent_scope_id="global", created_at=datetime(2026, 6, 10, tzinfo=UTC),
    ))
    candidates: dict[str, list[MemoryCandidate]] = {}

    def add_candidate(message_id: str, content: str, value: str) -> None:
        candidates[message_id] = [MemoryCandidate(
            content=content, memory_type=MemoryType.FACT, subject="alpha",
            predicate="uses_database", object=value, proposed_scope_level="scope",
            confidence=1.0, durability=1.0, source_message_ids=[message_id],
        )]

    add_candidate("message-correct", "Alpha uses MongoDB.", "MongoDB")
    add_candidate("message-correction", "Alpha uses MongoDB.", "MongoDB")
    add_candidate("message-error", "Alpha uses PostgreSQL.", "PostgreSQL")
    extractor = ScenarioExtractor(candidates)
    embedder = KeywordEmbeddingProvider()
    system = ScopeGraphMemorySystem(
        repository, extractor, retriever=ScopeAwareRetriever(repository, embedder)
    )
    started = datetime(2026, 6, 10, tzinfo=UTC)
    for session_id, message_id, content in (
        ("session-correct", "message-correct", "Alpha uses MongoDB."),
        ("session-error", "message-error", "Alpha uses PostgreSQL."),
    ):
        await system.ingest_session(
            SessionInput(
                id=session_id, scope_id="scope_alpha", started_at=started,
                messages=[SourceMessageCreate(
                    id=message_id, session_id=session_id, role=MessageRole.USER,
                    content=content, turn_index=0, timestamp=started,
                )],
            ),
            current_scope=ScopeRef(id="scope_alpha", name="Alpha", scope_type=ScopeType.PROJECT,
                                   session_id=session_id),
        )
    memories = await repository.list_memories(scope_id="scope_alpha", include_inactive=True)
    erroneous = next(memory for memory in memories if "PostgreSQL" in memory.content)
    if condition == "conversational":
        await system.ingest_session(
            SessionInput(
                id="session-conversation-correction", scope_id="scope_alpha", started_at=started,
                messages=[SourceMessageCreate(
                    id="message-correction", session_id="session-conversation-correction",
                    role=MessageRole.USER, content="Alpha uses MongoDB.", turn_index=0,
                    timestamp=started,
                )],
            ),
            current_scope=ScopeRef(id="scope_alpha", name="Alpha", scope_type=ScopeType.PROJECT,
                                   session_id="session-conversation-correction"),
        )
    elif condition == "graph":
        await system.corrections.edit(
            erroneous.id,
            MemoryEditRequest(
                content="Alpha uses MongoDB.", actor="eval", reason="direct correction"
            ),
        )
    relapse: list[str] = []
    probe_results: list[dict[str, object]] = []
    for count in range(1, max(probes) + 1):
        session_id = f"future-{count}"
        await system.ingest_session(
            SessionInput(
                id=session_id, scope_id="scope_alpha",
                started_at=started + timedelta(days=count),
            ),
            current_scope=ScopeRef(id="scope_alpha", name="Alpha", scope_type=ScopeType.PROJECT,
                                   session_id=session_id),
        )
        if count in probes:
            result = await system.retrieve(
                "What database does Alpha use?",
                current_scope=ScopeRef(id="scope_alpha", name="Alpha", scope_type=ScopeType.PROJECT,
                                       session_id=session_id),
                top_k=5, token_budget=100,
            )
            evidence = "\n".join(item.content for item in result.items)
            relapse.append(evidence)
            probe_results.append({"after_sessions": count, "evidence": evidence,
                                  "relapsed": "PostgreSQL" in evidence})
    return {"condition": condition, "probes": list(probes), "relapses": relapse,
            "probe_results": probe_results,
            "error_relapse_rate": error_relapse_rate(relapse, "PostgreSQL")}


async def run_correction_evaluation(
    probes: tuple[int, ...] = (1, 5, 10, 20),
) -> list[dict[str, object]]:
    return [
        await _condition(condition, probes)
        for condition in ("none", "conversational", "graph")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/raw/correction_persistence.jsonl")
    )
    args = parser.parse_args()
    results = asyncio.run(run_correction_evaluation())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(result) + "\n" for result in results))
    print(args.output)


if __name__ == "__main__":
    main()
