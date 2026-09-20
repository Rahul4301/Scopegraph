"""Run one deterministic dataset/system pair and write raw JSONL."""

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from evals.adapters.cross_scope_mem import CrossScopeMemAdapter
from evals.metrics.storage import logical_storage_stats
from evals.runners.providers import (
    EvaluationProviders,
    build_cross_scope_providers,
    freeze_extraction,
)
from evals.schemas import CrossScopeScenario, EvaluationRecord
from scopegraph.backends.flat_graph import FlatGraphMemory
from scopegraph.backends.scopegraph import ScopeGraphMemorySystem
from scopegraph.backends.two_level_graph import TwoLevelGraphMemory
from scopegraph.backends.vector_memory import VectorMemory
from scopegraph.config import get_settings
from scopegraph.graph.in_memory import InMemoryMemoryRepository
from scopegraph.llm.answering import AnswerModel, OpenAICompatibleAnswerer
from scopegraph.memory.retriever import RetrievalConfig, ScopeAwareRetriever
from scopegraph.models.scope import ScopeRef


def _config(path: str | Path | None) -> tuple[dict[str, Any], str]:
    if path is None:
        return {}, "default"
    raw = Path(path).read_bytes()
    return yaml.safe_load(raw) or {}, hashlib.sha256(raw).hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


async def _system(name: str, scenario: CrossScopeScenario,
                  providers: EvaluationProviders,
                  retrieval_config: RetrievalConfig | None = None):
    repository = InMemoryMemoryRepository()
    for scope in scenario.scopes:
        await repository.create_scope(scope)
    embedder = providers.embedder
    extractor = providers.extractor
    if name == "vector_memory":
        system = VectorMemory(repository, extractor, embedder, retrieval_config=retrieval_config)
    elif name == "flat_graph":
        system = FlatGraphMemory(repository, extractor, embedder, retrieval_config=retrieval_config)
    elif name == "two_level_graph":
        system = TwoLevelGraphMemory(
            repository, extractor, embedder, retrieval_config=retrieval_config
        )
    elif name == "scopegraph":
        system = ScopeGraphMemorySystem(
            repository, extractor,
            retriever=ScopeAwareRetriever(repository, embedder, retrieval_config),
        )
    else:
        raise ValueError(f"Unknown evaluation system: {name}")
    return system, repository


async def run_scenario(scenario: CrossScopeScenario, *, system_name: str, run_id: str,
                       config: dict[str, Any], config_hash: str,
                       providers: EvaluationProviders,
                       retrieval_config: RetrievalConfig | None = None,
                       answer_model: AnswerModel | None = None) -> list[EvaluationRecord]:
    system, repository = await _system(system_name, scenario, providers, retrieval_config)
    scope_by_id = {scope.id: scope for scope in scenario.scopes}
    source_scopes = {message.id: session.scope_id for session in scenario.sessions
                     for message in session.messages}
    pending = iter(sorted(scenario.sessions, key=lambda session: session.started_at))
    next_session = next(pending, None)
    retrieval = config.get("retrieval", {})
    top_k = int(retrieval.get("top_k", 8))
    token_budget = int(retrieval.get("token_budget", 1500))
    records: list[EvaluationRecord] = []
    examples = sorted(scenario.examples,
                      key=lambda item: item.timestamp or datetime.max.replace(tzinfo=UTC))
    for example in examples:
        # Replay events at the query's snapshot; future updates must not supersede
        # evidence that was still current when an earlier question was asked.
        while next_session is not None and (
            example.timestamp is None or
            max(message.timestamp for message in next_session.messages) <= example.timestamp
        ):
            scope = scope_by_id[next_session.scope_id]
            await system.ingest_session(
                next_session,
                current_scope=ScopeRef(id=scope.id, name=scope.name, scope_type=scope.scope_type,
                                       session_id=next_session.id),
            )
            next_session = next(pending, None)
        memories = await repository.list_memories(include_inactive=True)
        scope = scope_by_id.get(example.current_scope_id) if example.current_scope_id else None
        current_scope = ScopeRef(
            id=scope.id, name=scope.name, scope_type=scope.scope_type,
            session_id=example.current_session_id,
        ) if scope else None
        expected_ids = [memory.id for memory in memories
                        if set(memory.source_ids) & set(example.gold_source_ids)]
        preparation_started = time.perf_counter()
        await providers.embedder.embed([example.question, *(memory.content for memory in memories)])
        preparation_ms = (time.perf_counter() - preparation_started) * 1000
        started = time.perf_counter()
        result = await system.retrieve(
            example.question, current_scope=current_scope, top_k=top_k,
            token_budget=token_budget, now=example.timestamp,
        )
        if answer_model is not None:
            answer = await answer_model.generate(question=example.question, context=result.items)
        else:
            answer = None  # Offline retrieval tests cannot measure model answer quality.
        answer_ms = (time.perf_counter() - started) * 1000 - result.retrieval_latency_ms
        memories = await repository.list_memories(include_inactive=True)
        storage = logical_storage_stats(
            await system.stats(),
            serialized_memory_bytes=sum(len(memory.model_dump_json().encode())
                                        for memory in memories),
            embedding_count=sum(memory.embedding is not None for memory in memories),
        )
        records.append(EvaluationRecord(
            protocol_version="cross-scope-v2",
            answer_evaluated=answer_model is not None,
            embedding_preparation_ms=preparation_ms,
            latency_protocol="warm-embeddings/in-memory-repository",
            evaluation_mode=str(config.get("evaluation_mode", "offline")),
            run_id=run_id, dataset="cross_scope_mem", system=system_name,
            scenario_id=scenario.scenario_id, question_id=example.question_id,
            question_type=example.question_type, question=example.question,
            gold_answer=example.gold_answer, hypothesis=answer,
            current_scope_id=example.current_scope_id, gold_scope_ids=example.gold_scope_ids,
            gold_memory_ids=expected_ids,
            gold_source_ids=example.gold_source_ids,
            allowed_scope_ids=example.allowed_scope_ids,
            retrieved_source_ids=[item.source_ids for item in result.items],
            retrieved_origin_scope_ids=[sorted({source_scopes[source]
                for source in item.source_ids if source in source_scopes})
                for item in result.items],
            retrieved_contents=[item.content for item in result.items],
            retrieved_memory_ids=[item.memory_id for item in result.items],
            retrieved_scope_ids=[item.scope_id for item in result.items],
            retrieved_statuses=[item.status.value for item in result.items],
            retrieval_scores=[item.score for item in result.items],
            retrieval_latency_ms=result.retrieval_latency_ms, retrieved_tokens=result.token_count,
            answer=answer, answer_latency_ms=max(0.0, answer_ms),
            input_tokens=None, output_tokens=None,
            storage_stats=storage, trace=[step.model_dump(mode="json") for step in result.trace],
            config_hash=config_hash, git_commit=_git_commit(), seed=scenario.seed,
            timestamp=datetime.now(UTC),
        ))
    return records


async def run_evaluation(*, system_name: str, seed: int = 42, difficulty: int = 2,
                         scenario_count: int = 1, config_path: str | None = None,
                         output: str | None = None, ablation: str | None = None,
                         live_answer: bool = False, live_extraction: bool = False,
                         live_embeddings: bool = False, live: bool = False,
                         provider_bank: dict[str, EvaluationProviders] | None = None) -> Path:
    live_answer = live_answer or live
    live_extraction = live_extraction or live
    live_embeddings = live_embeddings or live
    config, config_hash = _config(config_path)
    config["evaluation_mode"] = (
        f"extraction={'live' if live_extraction else 'oracle'};"
        f"embeddings={'live' if live_embeddings else 'hash'};"
        f"answer={'live' if live_answer else 'not-evaluated'}"
    )
    retrieval_config_data = deepcopy(config.get("retrieval", {}))
    if ablation == "no_scope_weighting":
        weights = retrieval_config_data.setdefault("weights", {})
        weights["semantic"] = float(weights.get("semantic", 0.40)) + float(
            weights.get("scope", 0.25)
        )
        weights["scope"] = 0.0
    elif ablation == "no_graph_traversal":
        weights = retrieval_config_data.setdefault("weights", {})
        weights["semantic"] = float(weights.get("semantic", 0.40)) + float(
            weights.get("graph", 0.07)
        )
        weights["graph"] = 0.0
        retrieval_config_data["max_graph_hops"] = 0
        retrieval_config_data["max_expanded_nodes"] = 0
    elif ablation not in {None, "full"}:
        raise ValueError(f"Unsupported Phase 7 ablation: {ablation}")
    config_hash = hashlib.sha256(json.dumps({
        "config": config, "retrieval": retrieval_config_data, "difficulty": difficulty,
        "scenario_count": scenario_count, "protocol": "cross-scope-v2",
    }, sort_keys=True).encode()).hexdigest()
    retrieval_config = RetrievalConfig.from_config(retrieval_config_data)
    answer_model: AnswerModel | None = None
    if live_answer:
        settings = get_settings()
        answer_model = OpenAICompatibleAnswerer(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            model=settings.llm_model,
        )
    adapter = CrossScopeMemAdapter(seed=seed, difficulty=difficulty, scenario_count=scenario_count)
    stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')
    run_id = f"{stamp}_cross_scope_mem_{system_name}_{seed}"
    records: list[EvaluationRecord] = []
    for scenario in adapter.scenarios():
        providers = (provider_bank[scenario.scenario_id] if provider_bank else
                     build_cross_scope_providers(
                         scenario, live_extraction=live_extraction,
                         live_embeddings=live_embeddings,
                     ))
        try:
            if provider_bank is None:
                await freeze_extraction(scenario, providers)
            records.extend(await run_scenario(
                scenario,
                system_name=system_name,
                run_id=run_id,
                config=config,
                config_hash=config_hash,
                providers=providers,
                retrieval_config=retrieval_config,
                answer_model=answer_model,
            ))
        finally:
            if provider_bank is None:
                providers.close()
    destination = Path(output or config.get("output_root", "results/raw"))
    if destination.suffix != ".jsonl":
        destination = destination / f"{run_id}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(record.model_dump_json() + "\n" for record in records))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cross_scope_mem")
    parser.add_argument("--system", required=True)
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--scenario-count", type=int, default=1)
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--live-answer", action="store_true")
    parser.add_argument("--live-extraction", action="store_true")
    parser.add_argument("--live-embeddings", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.dataset != "cross_scope_mem":
        raise SystemExit(
            "run_eval supports cross_scope_mem; use run_external for external datasets"
        )
    print(asyncio.run(run_evaluation(system_name=args.system, seed=args.seed,
                                     difficulty=args.difficulty, scenario_count=args.scenario_count,
                                     config_path=args.config, output=args.output,
                                     ablation=args.ablation,
                                     live_answer=args.live_answer or args.live,
                                     live_extraction=args.live_extraction or args.live,
                                     live_embeddings=args.live_embeddings or args.live)))


if __name__ == "__main__":
    main()
