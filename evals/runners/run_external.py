"""Replay a validated external benchmark file through one memory backend."""

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from evals.adapters.cross_scope_mem import KeywordEmbeddingProvider
from evals.adapters.external import evidence_source_ids, session_inputs
from evals.adapters.registry import EXTERNAL_DATASETS, external_adapters
from evals.metrics.storage import logical_storage_stats
from evals.runners.checkpoint import RecordCheckpoint, source_fingerprint
from evals.schemas import EvaluationRecord
from scopegraph.backends.flat_graph import FlatGraphMemory
from scopegraph.backends.scopegraph import ScopeGraphMemorySystem
from scopegraph.backends.two_level_graph import TwoLevelGraphMemory
from scopegraph.backends.vector_memory import VectorMemory
from scopegraph.config import get_settings
from scopegraph.embeddings.base import EmbeddingProvider
from scopegraph.embeddings.cache import CachedEmbedder, SQLiteEmbeddingCache
from scopegraph.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from scopegraph.graph.in_memory import InMemoryMemoryRepository
from scopegraph.llm.answering import AnswerModel, OpenAICompatibleAnswerer
from scopegraph.llm.extraction import CandidateExtractor, LLMMemoryExtractor
from scopegraph.llm.openai_compatible import OpenAICompatibleLLM
from scopegraph.llm.transport import close_provider
from scopegraph.memory.retriever import RetrievalConfig, ScopeAwareRetriever
from scopegraph.models.memory import MemoryCandidate, MemoryType
from scopegraph.models.scope import ScopeCreate, ScopeRef, ScopeType


class TurnMemoryExtractor:
    """Credential-free extractor that preserves each external turn as evidence."""

    async def extract(self, messages, *, current_scope, existing_memories=None):
        del current_scope, existing_memories
        return [
            MemoryCandidate(
                content=message.content,
                memory_type=MemoryType.SUMMARY,
                proposed_scope_level="scope",
                confidence=0.5,
                durability=0.8,
                source_message_ids=[message.id],
            )
            for message in messages
        ]


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _config(path: str | None) -> tuple[dict, str]:
    if path is None:
        return {}, "default"
    raw = Path(path).read_bytes()
    return yaml.safe_load(raw) or {}, hashlib.sha256(raw).hexdigest()


async def _answer(model: AnswerModel | None, question: str, items) -> str | None:
    if model is None:
        return None
    return await model.generate(question=question, context=items)


def _usage(provider: object | None, key: str) -> int | None:
    return getattr(provider, "last_usage", {}).get(key)


async def run_external(
    *,
    dataset: str,
    path: str | Path,
    system_name: str,
    config_path: str | None = None,
    output: str | Path | None = None,
    limit: int | None = None,
    live_answer: bool = False,
    live_extraction: bool = False,
    live_embeddings: bool = False,
    live: bool = False,
    resume: bool = False,
) -> Path:
    live_answer = live_answer or live
    live_extraction = live_extraction or live
    live_embeddings = live_embeddings or live
    source_path = Path(path)
    adapter = external_adapters()[dataset]
    examples = adapter.load(source_path)[:limit]
    config, config_hash = _config(config_path)
    retrieval = RetrievalConfig.from_config(config.get("retrieval", {}))
    answer_model: AnswerModel | None = None
    if live_answer:
        settings = get_settings()
        answer_model = OpenAICompatibleAnswerer(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            model=settings.llm_model,
        )
    embedding_cache: SQLiteEmbeddingCache | None = None
    if live_embeddings:
        embedding_cache = SQLiteEmbeddingCache(get_settings().embedding_cache_path)
        settings = get_settings()
        embedder: EmbeddingProvider = CachedEmbedder(
            OpenAICompatibleEmbeddingProvider(
                base_url=settings.embedding_base_url,
                api_key=settings.embedding_api_key.get_secret_value(),
                model=settings.embedding_model,
            ),
            embedding_cache,
        )
    else:
        embedder = KeywordEmbeddingProvider()
    if live_extraction:
        settings = get_settings()
        extractor: CandidateExtractor = LLMMemoryExtractor(
            OpenAICompatibleLLM(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key.get_secret_value(),
                model=settings.llm_model,
            )
        )
    else:
        extractor = TurnMemoryExtractor()
    settings = get_settings()
    source_bytes = await asyncio.to_thread(source_path.read_bytes)
    protocol_payload = {
        "config_hash": config_hash,
        "dataset_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "dataset": dataset,
        "system": system_name,
        "limit": limit,
        "live_answer": live_answer,
        "live_extraction": live_extraction,
        "live_embeddings": live_embeddings,
        "llm_model": settings.llm_model if live_answer or live_extraction else None,
        "embedding_model": settings.embedding_model if live_embeddings else None,
        "source_fingerprint": source_fingerprint(),
    }
    config_hash = hashlib.sha256(
        json.dumps(protocol_payload, sort_keys=True).encode()
    ).hexdigest()
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{dataset}_{system_name}"
    destination = Path(output or config.get("output_root", "results/raw"))
    if destination.suffix != ".jsonl":
        destination /= f"{run_id}.jsonl"
    checkpoint = RecordCheckpoint(destination, resume=resume)
    for record in checkpoint.records.values():
        if record.config_hash != config_hash or record.system != system_name:
            raise ValueError(
                "Checkpoint does not match the dataset, code, config, models or system"
            )
    if checkpoint.records:
        run_id = next(iter(checkpoint.records.values())).run_id
    try:
        for example in examples:
            key = (example.example_id, example.question_id)
            if key in checkpoint.records:
                continue
            repository = InMemoryMemoryRepository()
            global_scope_id = f"external-global-{example.example_id}"
            scope_id = f"external-{dataset}-{example.example_id}"
            await repository.create_scope(
                ScopeCreate(id=global_scope_id, name="Global", scope_type=ScopeType.GLOBAL)
            )
            await repository.create_scope(
                ScopeCreate(
                    id=scope_id,
                    name=f"{dataset}:{example.example_id}",
                    scope_type=ScopeType.CUSTOM,
                    parent_scope_id=global_scope_id,
                )
            )
            if system_name == "vector_memory":
                system = VectorMemory(repository, extractor, embedder, retrieval_config=retrieval)
            elif system_name == "flat_graph":
                system = FlatGraphMemory(
                    repository, extractor, embedder, retrieval_config=retrieval
                )
            elif system_name == "two_level_graph":
                system = TwoLevelGraphMemory(
                    repository, extractor, embedder, retrieval_config=retrieval
                )
            elif system_name == "scopegraph":
                system = ScopeGraphMemorySystem(
                    repository,
                    extractor,
                    retriever=ScopeAwareRetriever(repository, embedder, retrieval),
                )
            else:
                raise ValueError(f"Unknown evaluation system: {system_name}")
            inputs = session_inputs(example, scope_id=scope_id, as_of=example.question_date)
            gold_sources = evidence_source_ids(example, inputs)
            for session in inputs:
                await system.ingest_session(
                    session,
                    current_scope=ScopeRef(
                        id=scope_id,
                        name=f"{dataset}:{example.example_id}",
                        scope_type=ScopeType.CUSTOM,
                        session_id=session.id,
                    ),
                )
            retrieval_cfg = config.get("retrieval", {})
            top_k = int(retrieval_cfg.get("top_k", 8))
            token_budget = int(retrieval_cfg.get("token_budget", 1500))
            memories = await repository.list_memories(include_inactive=True)
            preparation_started = time.perf_counter()
            await embedder.embed([example.question, *(memory.content for memory in memories)])
            preparation_ms = (time.perf_counter() - preparation_started) * 1000
            result = await system.retrieve(
                example.question,
                current_scope=ScopeRef(id=scope_id, scope_type=ScopeType.CUSTOM),
                top_k=top_k,
                token_budget=token_budget,
                now=example.question_date,
            )
            answer_started = time.perf_counter()
            answer = await _answer(answer_model, example.question, result.items)
            answer_latency = (
                (time.perf_counter() - answer_started) * 1000 if answer_model else None
            )
            stats = await system.stats()
            record = EvaluationRecord(
                protocol_version="external-v2",
                latency_protocol="warm-embeddings/in-memory-repository",
                embedding_preparation_ms=preparation_ms,
                run_id=run_id,
                    dataset=dataset,
                    system=system_name,
                    scenario_id=example.example_id,
                    question_id=example.question_id,
                    question_type=example.question_type,
                    question=example.question,
                    gold_answer=example.answer,
                    hypothesis=answer,
                    current_scope_id=scope_id,
                    gold_scope_ids=[scope_id] if gold_sources else [],
                    gold_source_ids=gold_sources,
                    allowed_scope_ids=[global_scope_id, scope_id],
                    retrieved_source_ids=[item.source_ids for item in result.items],
                    retrieved_origin_scope_ids=[
                        [scope_id] for _ in result.items
                    ],
                    retrieved_contents=[item.content for item in result.items],
                    retrieved_memory_ids=[item.memory_id for item in result.items],
                    retrieved_scope_ids=[item.scope_id for item in result.items],
                    retrieved_statuses=[item.status.value for item in result.items],
                    retrieval_scores=[item.score for item in result.items],
                    retrieval_latency_ms=result.retrieval_latency_ms,
                    retrieved_tokens=result.token_count,
                    answer=answer,
                    answer_evaluated=live_answer,
                    answer_latency_ms=answer_latency,
                    input_tokens=_usage(answer_model, "prompt_tokens"),
                    output_tokens=_usage(answer_model, "completion_tokens"),
                    evaluation_mode=(
                        f"extraction={'live' if live_extraction else 'turn-preserving'};"
                        f"embeddings={'live' if live_embeddings else 'hash'};"
                        f"answer={'live' if live_answer else 'not-evaluated'}"
                    ),
                    storage_stats=logical_storage_stats(stats),
                    trace=[step.model_dump(mode="json") for step in result.trace],
                    config_hash=config_hash,
                    git_commit=_git_commit(),
                    seed=42,
            )
            checkpoint.append(record)
    finally:
        await close_provider(extractor)
        await close_provider(embedder)
        await close_provider(answer_model)
        if embedding_cache is not None:
            embedding_cache.close()
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=EXTERNAL_DATASETS, required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument(
        "--system",
        choices=("vector_memory", "flat_graph", "two_level_graph", "scopegraph"),
        required=True,
    )
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--live-answer", action="store_true")
    parser.add_argument("--live-extraction", action="store_true")
    parser.add_argument("--live-embeddings", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(
        asyncio.run(
            run_external(
                dataset=args.dataset,
                path=args.path,
                system_name=args.system,
                config_path=args.config,
                output=args.output,
                limit=args.limit,
                live_answer=args.live_answer,
                live_extraction=args.live_extraction,
                live_embeddings=args.live_embeddings,
                live=args.live,
                resume=args.resume,
            )
        )
    )


if __name__ == "__main__":
    main()
