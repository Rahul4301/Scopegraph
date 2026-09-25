"""Run one deterministic dataset/system pair and write raw JSONL."""

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from evals.adapters.cross_scope_mem import CrossScopeMemAdapter
from evals.analysis.scope_classification import evaluate_scope_classification
from evals.metrics.storage import logical_storage_stats
from evals.runners.checkpoint import RecordCheckpoint, save_json, source_fingerprint
from evals.runners.providers import (
    EvaluationProviders,
    build_cross_scope_providers,
    freeze_extraction,
)
from evals.schemas import CrossScopeScenario, EvaluationRecord
from scopegraph.backends.scopegraph import ScopeGraphMemorySystem
from scopegraph.config import get_settings
from scopegraph.graph.client import Neo4jClient
from scopegraph.graph.in_memory import InMemoryMemoryRepository
from scopegraph.graph.repository import Neo4jMemoryRepository
from scopegraph.graph.schema import ensure_schema
from scopegraph.llm.answering import AnswerModel, OpenAICompatibleAnswerer
from scopegraph.llm.extraction import CandidateExtractor
from scopegraph.llm.transport import close_provider
from scopegraph.memory.retriever import RetrievalConfig, ScopeAwareRetriever
from scopegraph.models.memory import MemoryCandidate
from scopegraph.models.scope import ScopeRef, ScopeType
from scopegraph.models.source import SourceMessage


class TwoLevelControlExtractor:
    """Project durable memories into global, retaining only session/global levels."""

    def __init__(self, wrapped: CandidateExtractor) -> None:
        self.wrapped = wrapped

    async def extract(
        self,
        messages: list[SourceMessage],
        *,
        current_scope: ScopeRef | None,
        existing_memories: list[str] | None = None,
    ) -> list[MemoryCandidate]:
        candidates = await self.wrapped.extract(
            messages,
            current_scope=current_scope,
            existing_memories=existing_memories,
        )
        if current_scope is None or current_scope.scope_type is ScopeType.GLOBAL:
            return candidates
        return [
            candidate
            if candidate.proposed_scope_level == "session"
            else candidate.model_copy(
                update={"proposed_scope_level": "global", "explicit_global_signal": True}
            )
            for candidate in candidates
        ]


def _config(path: str | Path | None) -> tuple[dict[str, Any], str]:
    if path is None:
        return {}, "default"
    raw = Path(path).read_bytes()
    return yaml.safe_load(raw) or {}, hashlib.sha256(raw).hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


async def _system(
    name: str,
    scenario: CrossScopeScenario,
    providers: EvaluationProviders,
    retrieval_config: RetrievalConfig | None = None,
    *,
    storage: str = "memory",
    allow_neo4j_reset: bool = False,
    ablation: str = "full",
):
    client: Neo4jClient | None = None
    if storage == "memory":
        repository = InMemoryMemoryRepository()
    elif storage == "neo4j":
        if not allow_neo4j_reset:
            raise ValueError(
                "Neo4j evaluation clears the selected database between scenarios; "
                "pass --allow-neo4j-reset only for a disposable evaluation database"
            )
        client = Neo4jClient(get_settings())
        if not await client.health():
            await client.close()
            raise RuntimeError("Neo4j evaluation database is unavailable")
        await ensure_schema(client)
        await client.execute_write("MATCH (n) DETACH DELETE n")
        repository = Neo4jMemoryRepository(client)
    else:
        raise ValueError(f"Unsupported evaluation storage: {storage}")
    try:
        for scope in scenario.scopes:
            await repository.create_scope(scope)
        embedder = providers.embedder
        extractor: CandidateExtractor = providers.extractor
        if ablation == "two_level_control":
            extractor = TwoLevelControlExtractor(extractor)
        if name != "scopegraph":
            raise ValueError(f"Unsupported evaluation system: {name}")
        system = ScopeGraphMemorySystem(
            repository,
            extractor,
            retriever=ScopeAwareRetriever(repository, embedder, retrieval_config),
        )
    except BaseException:
        if client is not None:
            await client.close()
        raise
    return system, repository, client


async def run_scenario(
    scenario: CrossScopeScenario,
    *,
    system_name: str,
    run_id: str,
    config: dict[str, Any],
    config_hash: str,
    providers: EvaluationProviders,
    retrieval_config: RetrievalConfig | None = None,
    answer_model: AnswerModel | None = None,
    completed: set[str] | None = None,
    on_record: Callable[[EvaluationRecord], None] | None = None,
    on_progress: Callable[[EvaluationRecord], None] | None = None,
    storage: str = "memory",
    allow_neo4j_reset: bool = False,
    ablation: str = "full",
) -> list[EvaluationRecord]:
    system, repository, client = await _system(
        system_name,
        scenario,
        providers,
        retrieval_config,
        storage=storage,
        allow_neo4j_reset=allow_neo4j_reset,
        ablation=ablation,
    )
    scope_by_id = {scope.id: scope for scope in scenario.scopes}
    source_scopes = {
        message.id: session.scope_id
        for session in scenario.sessions
        for message in session.messages
    }
    pending = iter(sorted(scenario.sessions, key=lambda session: session.started_at))
    next_session = next(pending, None)
    retrieval = config.get("retrieval", {})
    top_k = int(retrieval.get("top_k", 8))
    token_budget = int(retrieval.get("token_budget", 1500))
    records: list[EvaluationRecord] = []
    examples = sorted(
        scenario.examples, key=lambda item: item.timestamp or datetime.max.replace(tzinfo=UTC)
    )
    try:
        for example in examples:
            # Replay events at the query's snapshot; future updates must not supersede
            # evidence that was still current when an earlier question was asked.
            while next_session is not None and (
                example.timestamp is None
                or max(message.timestamp for message in next_session.messages) <= example.timestamp
            ):
                scope = scope_by_id[next_session.scope_id]
                await system.ingest_session(
                    next_session,
                    current_scope=ScopeRef(
                        id=scope.id,
                        name=scope.name,
                        scope_type=scope.scope_type,
                        session_id=next_session.id,
                    ),
                )
                next_session = next(pending, None)
            if completed and example.question_id in completed:
                continue
            memories = await repository.list_memories(include_inactive=True)
            scope = scope_by_id.get(example.current_scope_id) if example.current_scope_id else None
            current_scope = (
                ScopeRef(
                    id=scope.id,
                    name=scope.name,
                    scope_type=scope.scope_type,
                    session_id=example.current_session_id,
                )
                if scope
                else None
            )
            expected_ids = [
                memory.id
                for memory in memories
                if set(memory.source_ids) & set(example.gold_source_ids)
            ]
            preparation_started = time.perf_counter()
            await providers.embedder.embed(
                [example.question, *(memory.content for memory in memories)]
            )
            preparation_ms = (time.perf_counter() - preparation_started) * 1000
            started = time.perf_counter()
            result = await system.retrieve(
                example.question,
                current_scope=current_scope,
                top_k=top_k,
                token_budget=token_budget,
                now=example.timestamp,
            )
            if answer_model is not None:
                answer = await answer_model.generate(
                    question=example.question, context=result.items
                )
            else:
                answer = None  # Offline retrieval tests cannot measure model answer quality.
            answer_ms = (time.perf_counter() - started) * 1000 - result.retrieval_latency_ms
            memories = await repository.list_memories(include_inactive=True)
            storage_stats = logical_storage_stats(
                await system.stats(),
                serialized_memory_bytes=sum(
                    len(memory.model_dump_json().encode()) for memory in memories
                ),
                embedding_count=sum(memory.embedding is not None for memory in memories),
            )
            records.append(
                EvaluationRecord(
                    protocol_version=f"cross-scope-v4/{scenario.profile}",
                    answer_evaluated=answer_model is not None,
                    embedding_preparation_ms=preparation_ms,
                    latency_protocol=f"warm-embeddings/{storage}-repository",
                    evaluation_mode=str(config.get("evaluation_mode", "offline")),
                    run_id=run_id,
                    dataset="cross_scope_mem",
                    system=system_name,
                    ablation=ablation,
                    scenario_id=scenario.scenario_id,
                    question_id=example.question_id,
                    question_type=example.question_type,
                    question=example.question,
                    gold_answer=example.gold_answer,
                    hypothesis=answer,
                    current_scope_id=example.current_scope_id,
                    gold_scope_ids=example.gold_scope_ids,
                    gold_memory_ids=expected_ids,
                    gold_source_ids=example.gold_source_ids,
                    allowed_scope_ids=example.allowed_scope_ids,
                    retrieved_source_ids=[item.source_ids for item in result.items],
                    retrieved_origin_scope_ids=[
                        sorted(
                            {
                                source_scopes[source]
                                for source in item.source_ids
                                if source in source_scopes
                            }
                        )
                        for item in result.items
                    ],
                    retrieved_contents=[item.content for item in result.items],
                    retrieved_memory_ids=[item.memory_id for item in result.items],
                    retrieved_scope_ids=[item.scope_id for item in result.items],
                    retrieved_statuses=[item.status.value for item in result.items],
                    retrieval_scores=[item.score for item in result.items],
                    retrieval_latency_ms=result.retrieval_latency_ms,
                    retrieved_tokens=result.token_count,
                    answer=answer,
                    answer_latency_ms=max(0.0, answer_ms),
                    input_tokens=getattr(answer_model, "last_usage", {}).get("prompt_tokens"),
                    output_tokens=getattr(answer_model, "last_usage", {}).get("completion_tokens"),
                    storage_stats=storage_stats,
                    trace=[step.model_dump(mode="json") for step in result.trace],
                    config_hash=config_hash,
                    git_commit=_git_commit(),
                    seed=scenario.seed,
                    timestamp=datetime.now(UTC),
                )
            )
            if on_record is not None:
                on_record(records[-1])
            if on_progress is not None:
                on_progress(records[-1])
    finally:
        if client is not None:
            try:
                await client.execute_write("MATCH (n) DETACH DELETE n")
            finally:
                await client.close()
    return records


async def run_evaluation(
    *,
    system_name: str,
    seed: int = 42,
    difficulty: int = 2,
    scenario_count: int = 1,
    config_path: str | None = None,
    output: str | None = None,
    ablation: str | None = None,
    live_answer: bool = False,
    live_extraction: bool = False,
    live_embeddings: bool = False,
    live: bool = False,
    provider_bank: dict[str, EvaluationProviders] | None = None,
    resume: bool = False,
    profile: str = "research",
    on_progress: Callable[[EvaluationRecord], None] | None = None,
    storage: str = "memory",
    allow_neo4j_reset: bool = False,
) -> Path:
    live_answer = live_answer or live
    live_extraction = live_extraction or live
    live_embeddings = live_embeddings or live
    config, config_hash = _config(config_path)
    config["evaluation_mode"] = (
        f"extraction={'live' if live_extraction else 'oracle'};"
        f"embeddings={'live' if live_embeddings else 'hash'};"
        f"answer={'live' if live_answer else 'not-evaluated'};storage={storage}"
    )
    retrieval_config_data = deepcopy(config.get("retrieval", {}))
    effective_ablation = ablation or "full"
    if effective_ablation in {"no_scope_hierarchy", "flat_graph_control"}:
        weights = retrieval_config_data.setdefault("weights", {})
        weights["semantic"] = float(weights.get("semantic", 0.40)) + float(
            weights.get("scope", 0.25)
        )
        weights["scope"] = 0.0
        retrieval_config_data["scope_access_mode"] = "all"
        retrieval_config_data["enforce_session_access"] = False
    elif effective_ablation == "no_graph_traversal":
        weights = retrieval_config_data.setdefault("weights", {})
        weights["semantic"] = float(weights.get("semantic", 0.40)) + float(
            weights.get("graph", 0.07)
        )
        weights["graph"] = 0.0
        retrieval_config_data["max_graph_hops"] = 0
        retrieval_config_data["max_expanded_nodes"] = 0
    elif effective_ablation == "no_temporal_status":
        weights = retrieval_config_data.setdefault("weights", {})
        weights["semantic"] = float(weights.get("semantic", 0.40)) + float(
            weights.get("temporal", 0.15)
        )
        weights["temporal"] = 0.0
        retrieval_config_data["enforce_temporal_status"] = False
    elif effective_ablation == "vector_only_control":
        retrieval_config_data["weights"] = {
            "semantic": 1.0,
            "scope": 0.0,
            "temporal": 0.0,
            "confidence": 0.0,
            "graph": 0.0,
            "recency": 0.0,
        }
        retrieval_config_data["scope_access_mode"] = "all"
        retrieval_config_data["enforce_session_access"] = False
        retrieval_config_data["max_graph_hops"] = 0
        retrieval_config_data["max_expanded_nodes"] = 0
    elif effective_ablation == "two_level_control":
        pass
    elif effective_ablation != "full":
        raise ValueError(f"Unsupported cross-scope ablation: {effective_ablation}")
    settings = get_settings()
    config_hash = hashlib.sha256(
        json.dumps(
            {
                "config": config,
                "retrieval": retrieval_config_data,
                "difficulty": difficulty,
                "scenario_count": scenario_count,
                "protocol": f"cross-scope-v4/{profile}",
                "ablation": effective_ablation,
                "storage": storage,
                "source_fingerprint": source_fingerprint(),
                "models": {
                    "llm": settings.llm_model if live_answer or live_extraction else None,
                    "embeddings": settings.embedding_model if live_embeddings else None,
                },
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    retrieval_config = RetrievalConfig.from_config(retrieval_config_data)
    answer_model: AnswerModel | None = None
    if live_answer:
        settings = get_settings()
        answer_model = OpenAICompatibleAnswerer(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            model=settings.llm_model,
            max_output_tokens=int(config.get("models", {}).get("max_output_tokens", 300)),
        )
    adapter = CrossScopeMemAdapter(
        seed=seed, difficulty=difficulty, scenario_count=scenario_count, profile=profile
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{stamp}_cross_scope_mem_{system_name}_{seed}"
    destination = Path(output or config.get("output_root", "results/raw"))
    if destination.suffix != ".jsonl":
        destination = destination / f"{run_id}.jsonl"
    checkpoint = RecordCheckpoint(destination, resume=resume)
    for record in checkpoint.records.values():
        if record.config_hash != config_hash or record.system != system_name:
            raise ValueError("Checkpoint does not match current code, config, models or system")
    if checkpoint.records:
        run_id = next(iter(checkpoint.records.values())).run_id
    scenarios = adapter.scenarios()
    standalone_extractions: dict[str, dict[str, list[MemoryCandidate]]] = {}
    extraction_path = destination.with_suffix(".extractions.json")
    frozen_extractions = (
        json.loads(extraction_path.read_text())
        if resume and extraction_path.exists() and provider_bank is None
        else {}
    )
    for scenario in scenarios:
        completed = {
            question_id
            for scenario_id, question_id in checkpoint.records
            if scenario_id == scenario.scenario_id
        }
        scenario_complete = completed == {example.question_id for example in scenario.examples}
        if scenario_complete and provider_bank is not None:
            continue
        providers = (
            provider_bank[scenario.scenario_id]
            if provider_bank
            else build_cross_scope_providers(
                scenario,
                live_extraction=live_extraction,
                live_embeddings=live_embeddings,
            )
        )
        try:
            if provider_bank is None:

                def persist(candidates, scenario_id=scenario.scenario_id):
                    frozen_extractions[scenario_id] = {
                        message_id: [candidate.model_dump(mode="json") for candidate in values]
                        for message_id, values in candidates.items()
                    }
                    save_json(extraction_path, frozen_extractions)

                candidates = await freeze_extraction(
                    scenario,
                    providers,
                    cached={
                        message_id: [
                            MemoryCandidate.model_validate(candidate) for candidate in values
                        ]
                        for message_id, values in frozen_extractions.get(
                            scenario.scenario_id, {}
                        ).items()
                    },
                    checkpoint=persist,
                )
                standalone_extractions[scenario.scenario_id] = candidates
                persist(candidates)
            if scenario_complete:
                continue
            await run_scenario(
                scenario,
                system_name=system_name,
                run_id=run_id,
                config=config,
                config_hash=config_hash,
                providers=providers,
                retrieval_config=retrieval_config,
                answer_model=answer_model,
                completed=completed,
                on_record=checkpoint.append,
                on_progress=on_progress,
                storage=storage,
                allow_neo4j_reset=allow_neo4j_reset,
                ablation=effective_ablation,
            )
        finally:
            if provider_bank is None:
                await providers.aclose()
    if provider_bank is None:
        classification = evaluate_scope_classification(
            scenarios,
            standalone_extractions,
            live_extraction=live_extraction,
        )
        save_json(
            destination.with_suffix(".classification.json"),
            classification.model_dump(mode="json"),
        )
    await close_provider(answer_model)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cross_scope_mem")
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--scenario-count", type=int, default=1)
    parser.add_argument(
        "--ablation",
        choices=[
            "full",
            "no_scope_hierarchy",
            "flat_graph_control",
            "no_graph_traversal",
            "no_temporal_status",
            "vector_only_control",
            "two_level_control",
        ],
        default="full",
    )
    parser.add_argument("--live-answer", action="store_true")
    parser.add_argument("--live-extraction", action="store_true")
    parser.add_argument("--live-embeddings", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile", choices=["smoke", "research"], default="research")
    parser.add_argument(
        "--allow-neo4j-reset",
        action="store_true",
        help="allow clearing the configured Neo4j database between scenarios",
    )
    args = parser.parse_args()
    if args.dataset != "cross_scope_mem":
        raise SystemExit(
            "run_eval supports cross_scope_mem; use run_external for external datasets"
        )

    def report(record: EvaluationRecord) -> None:
        print(
            f"[{record.system}] {record.scenario_id} | {record.question_id} | "
            f"retrieval={record.retrieval_latency_ms:.1f}ms",
            flush=True,
        )

    print(
        asyncio.run(
            run_evaluation(
                system_name="scopegraph",
                seed=args.seed,
                difficulty=args.difficulty,
                scenario_count=args.scenario_count,
                config_path=args.config,
                output=args.output,
                ablation=args.ablation,
                resume=args.resume,
                profile=args.profile,
                live_answer=args.live_answer or args.live,
                live_extraction=args.live_extraction or args.live,
                live_embeddings=args.live_embeddings or args.live,
                storage="neo4j",
                allow_neo4j_reset=args.allow_neo4j_reset,
                on_progress=report,
            )
        )
    )


if __name__ == "__main__":
    main()
