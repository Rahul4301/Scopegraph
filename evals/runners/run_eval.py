"""Run one deterministic dataset/system pair and write raw JSONL."""

import argparse
import asyncio
import hashlib
import re
import subprocess
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from evals.adapters.cross_scope_mem import CrossScopeMemAdapter, KeywordEmbeddingProvider
from evals.metrics.storage import logical_storage_stats
from evals.schemas import CrossScopeScenario, EvaluationRecord
from scopegraph.backends.flat_graph import FlatGraphMemory
from scopegraph.backends.scopegraph import ScopeGraphMemorySystem
from scopegraph.backends.two_level_graph import TwoLevelGraphMemory
from scopegraph.backends.vector_memory import VectorMemory
from scopegraph.graph.in_memory import InMemoryMemoryRepository
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
                  retrieval_config: RetrievalConfig | None = None):
    repository = InMemoryMemoryRepository()
    for scope in scenario.scopes:
        await repository.create_scope(scope)
    embedder = KeywordEmbeddingProvider()
    extractor = CrossScopeMemAdapter.extractor(scenario)
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
                       retrieval_config: RetrievalConfig | None = None) -> list[EvaluationRecord]:
    system, repository = await _system(system_name, scenario, retrieval_config)
    scope_by_id = {scope.id: scope for scope in scenario.scopes}
    for session in scenario.sessions:
        scope = scope_by_id[session.scope_id]
        await system.ingest_session(
            session,
            current_scope=ScopeRef(id=scope.id, name=scope.name, scope_type=scope.scope_type,
                                   session_id=session.id),
        )
    memories = await repository.list_memories(include_inactive=True)
    memory_by_content = {memory.content.casefold(): memory.id for memory in memories}
    stats = await system.stats()
    storage = logical_storage_stats(
        stats,
        serialized_memory_bytes=sum(len(memory.content.encode()) for memory in memories),
        embedding_count=sum(memory.embedding is not None for memory in memories),
    )
    retrieval = config.get("retrieval", {})
    top_k = int(retrieval.get("top_k", 8))
    token_budget = int(retrieval.get("token_budget", 1500))
    records: list[EvaluationRecord] = []
    for example in scenario.examples:
        scope = scope_by_id.get(example.current_scope_id) if example.current_scope_id else None
        current_scope = ScopeRef(
            id=scope.id, name=scope.name, scope_type=scope.scope_type,
            session_id=example.current_session_id,
        ) if scope else None
        expected_ids = [memory_by_content[content.casefold()]
                        for content in example.gold_memory_contents
                        if content.casefold() in memory_by_content]
        started = time.perf_counter()
        result = await system.retrieve(
            example.question, current_scope=current_scope, top_k=top_k,
            token_budget=token_budget, now=example.timestamp,
        )
        answer = _deterministic_answer(result.items[0].content) if result.items else None
        answer_ms = (time.perf_counter() - started) * 1000 - result.retrieval_latency_ms
        records.append(EvaluationRecord(
            run_id=run_id, dataset="cross_scope_mem", system=system_name,
            scenario_id=scenario.scenario_id, question_id=example.question_id,
            question_type=example.question_type, question=example.question,
            gold_answer=example.gold_answer, hypothesis=answer,
            current_scope_id=example.current_scope_id, gold_scope_ids=example.gold_scope_ids,
            gold_memory_ids=expected_ids,
            retrieved_memory_ids=[item.memory_id for item in result.items],
            retrieved_scope_ids=[item.scope_id for item in result.items],
            retrieved_statuses=[item.status.value for item in result.items],
            retrieval_scores=[item.score for item in result.items],
            retrieval_latency_ms=result.retrieval_latency_ms, retrieved_tokens=result.token_count,
            answer=answer, answer_latency_ms=max(0.0, answer_ms),
            input_tokens=len(example.question.split()),
            output_tokens=len(answer.split()) if answer else 0,
            storage_stats=storage, trace=[step.model_dump(mode="json") for step in result.trace],
            config_hash=config_hash, git_commit=_git_commit(), seed=scenario.seed,
            timestamp=datetime.now(UTC),
        ))
    return records


def _deterministic_answer(content: str) -> str:
    """Extract a short fact value without an external answer model."""
    patterns = (
        r"(?:uses|use)\s+([A-Z][A-Za-z0-9_-]+)",
        r"prefer\s+([A-Z][A-Za-z0-9_-]+)",
        r"database\s+to\s+([A-Z][A-Za-z0-9_-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return match.group(1)
    return content


async def run_evaluation(*, system_name: str, seed: int = 42, difficulty: int = 2,
                         scenario_count: int = 1, config_path: str | None = None,
                         output: str | None = None, ablation: str | None = None) -> Path:
    config, config_hash = _config(config_path)
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
    retrieval_config = RetrievalConfig.from_config(retrieval_config_data)
    adapter = CrossScopeMemAdapter(seed=seed, difficulty=difficulty, scenario_count=scenario_count)
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_cross_scope_mem_{system_name}_{seed}"
    records = [record for scenario in adapter.scenarios()
               for record in await run_scenario(scenario, system_name=system_name, run_id=run_id,
                                                config=config, config_hash=config_hash,
                                                retrieval_config=retrieval_config)]
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
    args = parser.parse_args()
    if args.dataset != "cross_scope_mem":
        raise SystemExit("Phase 7 currently implements only --dataset cross_scope_mem")
    print(asyncio.run(run_evaluation(system_name=args.system, seed=args.seed,
                                     difficulty=args.difficulty, scenario_count=args.scenario_count,
                                     config_path=args.config, output=args.output,
                                     ablation=args.ablation)))


if __name__ == "__main__":
    main()
