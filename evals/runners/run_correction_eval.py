"""Run the three-condition correction-persistence experiment."""

import argparse
import asyncio
import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

from evals.adapters.cross_scope_mem import KeywordEmbeddingProvider, ScenarioExtractor
from evals.metrics.correction_persistence import error_relapse_rate
from scopegraph.backends.scopegraph import ScopeGraphMemorySystem
from scopegraph.config import get_settings
from scopegraph.graph.client import Neo4jClient
from scopegraph.graph.repository import Neo4jMemoryRepository
from scopegraph.graph.schema import ensure_schema
from scopegraph.memory.retriever import ScopeAwareRetriever
from scopegraph.models.correction import MemoryEditRequest
from scopegraph.models.memory import MemoryCandidate, MemoryType
from scopegraph.models.scope import ScopeCreate, ScopeRef, ScopeType
from scopegraph.models.session import SessionInput
from scopegraph.models.source import MessageRole, SourceMessageCreate


async def _condition(
    condition: str, probes: tuple[int, ...], case_id: int
) -> dict[str, object]:
    client = Neo4jClient(get_settings())
    if not await client.health():
        await client.close()
        raise RuntimeError("Neo4j evaluation database is unavailable")
    await ensure_schema(client)
    await client.execute_write("MATCH (n) DETACH DELETE n")
    repository = Neo4jMemoryRepository(client)
    await repository.create_scope(ScopeCreate(
        id="global", name="Global", scope_type=ScopeType.GLOBAL,
        created_at=datetime(2026, 6, 10, tzinfo=UTC),
    ))
    await repository.create_scope(ScopeCreate(
        id="scope_alpha", name="Alpha", scope_type=ScopeType.PROJECT,
        parent_scope_id="global", created_at=datetime(2026, 6, 10, tzinfo=UTC),
    ))
    candidates: dict[str, list[MemoryCandidate]] = {}

    def add_candidate(
        message_id: str,
        content: str,
        value: str,
        *,
        predicate: str = "uses_database",
    ) -> None:
        candidates[message_id] = [MemoryCandidate(
            content=content, memory_type=MemoryType.FACT, subject="alpha",
            predicate=predicate, object=value, proposed_scope_level="scope",
            confidence=1.0, durability=1.0, source_message_ids=[message_id],
        )]

    fact_pairs = [
        ("MongoDB", "PostgreSQL"),
        ("Neo4j", "Redis"),
        ("SQLite", "MySQL"),
        ("Datomic", "PostgreSQL"),
    ]
    correct_value, error_value = fact_pairs[case_id % len(fact_pairs)]
    correct_content = f"Alpha uses {correct_value}."
    error_content = f"Alpha uses {error_value}."
    add_candidate("message-correct", correct_content, correct_value)
    add_candidate("message-correction", correct_content, correct_value)
    add_candidate("message-error", error_content, error_value)
    for count in range(1, max(probes) + 1):
        add_candidate(
            f"future-message-{count}",
            f"Alpha case {case_id} completed checkpoint {count}.",
            f"checkpoint-{case_id}-{count}",
            predicate=f"completed_checkpoint_{count}",
        )
    extractor = ScenarioExtractor(candidates)
    embedder = KeywordEmbeddingProvider()
    system = ScopeGraphMemorySystem(
        repository, extractor, retriever=ScopeAwareRetriever(repository, embedder)
    )
    started = datetime(2026, 6, 10, tzinfo=UTC)
    for session_id, message_id, content in (
        ("session-correct", "message-correct", correct_content),
        ("session-error", "message-error", error_content),
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
    erroneous = next(memory for memory in memories if error_value in memory.content)
    if condition == "conversational":
        await system.ingest_session(
            SessionInput(
                id="session-conversation-correction", scope_id="scope_alpha", started_at=started,
                messages=[SourceMessageCreate(
                    id="message-correction", session_id="session-conversation-correction",
                    role=MessageRole.USER, content=correct_content, turn_index=0,
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
                content=correct_content, actor="eval", reason="direct correction"
            ),
        )
    relapse: list[str] = []
    probe_results: list[dict[str, object]] = []
    for count in range(1, max(probes) + 1):
        session_id = f"future-{count}"
        message_id = f"future-message-{count}"
        content = f"Alpha case {case_id} completed checkpoint {count}."
        await system.ingest_session(
            SessionInput(
                id=session_id, scope_id="scope_alpha",
                started_at=started + timedelta(days=count),
                messages=[SourceMessageCreate(
                    id=message_id,
                    session_id=session_id,
                    role=MessageRole.USER,
                    content=content,
                    turn_index=0,
                    timestamp=started + timedelta(days=count),
                )],
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
                                  "relapsed": error_value in evidence})
    result = {"case_id": case_id, "condition": condition, "probes": list(probes),
              "correct_value": correct_value, "error_value": error_value,
              "relapses": relapse,
              "probe_results": probe_results,
              "error_relapse_rate": error_relapse_rate(relapse, error_value)}
    try:
        await client.execute_write("MATCH (n) DETACH DELETE n")
    finally:
        await client.close()
    return result


async def run_correction_evaluation(
    probes: tuple[int, ...] = (1, 5, 10, 20),
    cases: int = 30,
) -> list[dict[str, object]]:
    if cases < 1:
        raise ValueError("cases must be positive")
    return [
        await _condition(condition, probes, case_id)
        for case_id in range(cases)
        for condition in ("none", "conversational", "graph")
    ]


def summarize(results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    by_condition: dict[str, list[dict[str, object]]] = {}
    for result in results:
        by_condition.setdefault(str(result["condition"]), []).append(result)
    summary: dict[str, dict[str, float]] = {}
    generator = random.Random(42)
    for condition, rows in sorted(by_condition.items()):
        case_rates = [float(row["error_relapse_rate"]) for row in rows]
        bootstrapped = sorted(
            sum(case_rates[generator.randrange(len(case_rates))] for _ in case_rates)
            / len(case_rates)
            for _ in range(10_000)
        )
        summary[condition] = {
            "case_count": float(len(rows)),
            "error_relapse_rate": sum(case_rates) / len(case_rates),
            "error_relapse_rate_ci95_low": bootstrapped[250],
            "error_relapse_rate_ci95_high": bootstrapped[9750],
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/raw/correction_persistence.jsonl")
    )
    parser.add_argument("--cases", type=int, default=30)
    parser.add_argument(
        "--allow-neo4j-reset",
        action="store_true",
        help="allow clearing the configured evaluation-only Neo4j database",
    )
    args = parser.parse_args()
    if not args.allow_neo4j_reset:
        raise SystemExit(
            "Correction evaluation clears Neo4j; pass --allow-neo4j-reset only "
            "for a disposable evaluation database"
        )
    results = asyncio.run(run_correction_evaluation(cases=args.cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(result) + "\n" for result in results))
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summarize(results), indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(summary_path)


if __name__ == "__main__":
    main()
