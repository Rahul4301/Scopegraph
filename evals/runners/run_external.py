"""Replay a validated external benchmark file through ScopeGraph on Neo4j."""

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import yaml

from evals.adapters.cross_scope_mem import KeywordEmbeddingProvider, ScenarioExtractor
from evals.adapters.external import evidence_source_ids, session_inputs
from evals.adapters.registry import EXTERNAL_DATASETS, external_adapters
from evals.judges import OfficialBenchmarkJudge
from evals.metrics.official import official_score
from evals.metrics.storage import logical_storage_stats
from evals.runners.checkpoint import RecordCheckpoint, save_json, source_fingerprint
from evals.schemas import EvaluationRecord, ExternalBenchmarkExample
from scopegraph.backends.scopegraph import ScopeGraphMemorySystem
from scopegraph.config import get_settings
from scopegraph.embeddings.base import EmbeddingProvider
from scopegraph.embeddings.cache import CachedEmbedder, SQLiteEmbeddingCache
from scopegraph.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from scopegraph.graph.client import Neo4jClient
from scopegraph.graph.in_memory import InMemoryMemoryRepository
from scopegraph.graph.repository import Neo4jMemoryRepository
from scopegraph.graph.schema import ensure_schema
from scopegraph.llm.answering import AnswerModel, OpenAICompatibleAnswerer
from scopegraph.llm.extraction import CandidateExtractor, LLMMemoryExtractor
from scopegraph.llm.openai_compatible import OpenAICompatibleLLM
from scopegraph.llm.transport import close_provider
from scopegraph.memory.retriever import RetrievalConfig, ScopeAwareRetriever
from scopegraph.models.memory import MemoryCandidate, MemoryType
from scopegraph.models.scope import ScopeCreate, ScopeRef, ScopeType
from scopegraph.models.source import SourceMessage


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


async def _freeze_external_extraction(
    inputs,
    extractor: CandidateExtractor,
    *,
    scope: ScopeRef,
    cached: dict[str, list[MemoryCandidate]],
    checkpoint,
) -> dict[str, list[MemoryCandidate]]:
    """Extract each source session once, independent of mutable graph state."""
    by_message = dict(cached)
    for session in inputs:
        if not session.messages or all(message.id in by_message for message in session.messages):
            continue
        messages = [SourceMessage(**message.model_dump()) for message in session.messages]
        candidates = await extractor.extract(
            messages,
            current_scope=scope.model_copy(update={"session_id": session.id}),
            existing_memories=[],
        )
        message_ids = {message.id for message in session.messages}
        invalid = sorted(
            {
                source_id
                for candidate in candidates
                for source_id in candidate.source_message_ids
                if source_id not in message_ids
            }
        )
        if invalid:
            raise ValueError(
                f"Extractor returned source IDs outside session {session.id}: {invalid}"
            )
        for message in session.messages:
            by_message[message.id] = [
                candidate
                for candidate in candidates
                if message.id in candidate.source_message_ids
            ]
        checkpoint(by_message)
    return by_message


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


def _source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    auxiliary = path / "entity2id.json" if path.is_dir() else None
    if auxiliary is not None and auxiliary.is_file():
        paths.append(auxiliary)
    for item in paths:
        digest.update(item.name.encode())
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _answer_protocol(example: ExternalBenchmarkExample) -> tuple[str, int]:
    source = str(example.metadata.get("source", ""))
    if "infbench" in source:
        return "Summarize the book using only the retrieved evidence.", 2000
    if "recsys" in source:
        return (
            "Act as a movie recommender. Return 20 numbered movie recommendations and no extra "
            "sentences.",
            600,
        )
    if "eventqa" in source:
        return "Complete the task; return only the event that happens next.", 300
    if "icl" in source:
        return 'Return only "label: {label}" with the numerical label.', 100
    if "factconsolidation" in source:
        return (
            "Use the newest numbered fact when facts conflict. Return only the concise answer.",
            300,
        )
    if "detective" in source:
        return "Follow the strict output format in the question using only retrieved evidence.", 300
    return (
        "Answer from the relevant history as concisely as possible, using one phrase when "
        "possible.",
        300,
    )


async def _answer(
    model: AnswerModel | None, example: ExternalBenchmarkExample, items
) -> str | None:
    if model is None:
        return None
    instruction, max_output_tokens = _answer_protocol(example)
    return await model.generate(
        question=example.question,
        context=items,
        instruction=instruction,
        max_output_tokens=max_output_tokens,
    )


def _usage(provider: object | None, key: str) -> int | None:
    return getattr(provider, "last_usage", {}).get(key)


def _total_usage(provider: object | None) -> dict[str, int]:
    if provider is None:
        return {}
    transport = getattr(provider, "transport", None)
    if transport is not None:
        return dict(getattr(transport, "total_usage", {}))
    nested = getattr(provider, "provider", None)
    return _total_usage(nested)


def _usage_delta(before: dict[str, int], after: dict[str, int], prefix: str) -> dict[str, int]:
    return {
        f"{prefix}_{key}": value - before.get(key, 0)
        for key, value in after.items()
        if value - before.get(key, 0)
    }


def _corpus_id(dataset: str, example: ExternalBenchmarkExample) -> str:
    """Return the history identity shared by one or more official questions."""
    if "corpus_id" in example.metadata:
        return str(example.metadata["corpus_id"])
    if dataset == "locomo":
        return str(example.metadata["sample_id"])
    return example.example_id


def _failure_metric(dataset: str, example: ExternalBenchmarkExample) -> str:
    if dataset == "locomo":
        return "locomo_f1"
    if dataset == "longmemeval" or "longmemeval" in example.question_type:
        return "llm_judge_accuracy"
    if "infbench" in example.question_type:
        return "gpt-4-f1"
    if "recsys" in example.question_type:
        return "recsys_recall@5"
    if "eventqa" in example.question_type:
        return "substring_exact_match"
    if "ruler" in example.question_type:
        return "ruler_recall"
    if "factconsolidation" in example.question_type:
        return "substring_exact_match"
    return "exact_match"


def _failure_record(
    *,
    run_id: str,
    dataset: str,
    system_name: str,
    ablation: str,
    corpus_id: str,
    example: ExternalBenchmarkExample,
    config_hash: str,
    storage: str,
    live_answer: bool,
    live_extraction: bool,
    live_embeddings: bool,
    error: Exception,
) -> EvaluationRecord:
    scope_id = f"external-{dataset}-{corpus_id}"
    return EvaluationRecord(
        protocol_version="external-v4",
        latency_protocol=f"warm-embeddings/{storage}-repository",
        run_id=run_id,
        dataset=dataset,
        system=system_name,
        ablation=ablation,
        scenario_id=corpus_id,
        question_id=example.question_id,
        question_type=example.question_type,
        question=example.question,
        gold_answer=example.answer,
        current_scope_id=scope_id,
        allowed_scope_ids=[f"external-global-{corpus_id}", scope_id],
        answer_evaluated=live_answer,
        benchmark_metadata=example.metadata,
        official_metric=_failure_metric(dataset, example),
        official_score=0.0,
        failure_type=type(error).__name__,
        failure_message=str(error),
        evaluation_mode=(
            f"extraction={'live' if live_extraction else 'turn-preserving'};"
            f"embeddings={'live' if live_embeddings else 'hash'};"
            f"answer={'live' if live_answer else 'not-evaluated'};storage={storage}"
        ),
        config_hash=config_hash,
        git_commit=_git_commit(),
        seed=42,
    )


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
    storage: str = "neo4j",
    allow_neo4j_reset: bool = False,
    ablation: str = "full",
    extraction_cache: str | Path | None = None,
) -> Path:
    live_answer = live_answer or live
    live_extraction = live_extraction or live
    live_embeddings = live_embeddings or live
    source_path = Path(path)
    adapter = external_adapters()[dataset]
    examples = adapter.load(source_path)[:limit]
    config, config_hash = _config(config_path)
    retrieval_config_data = deepcopy(config.get("retrieval", {}))
    if ablation != "full":
        raise ValueError("Official external benchmarks run full ScopeGraph only")
    retrieval = RetrievalConfig.from_config(retrieval_config_data)
    answer_model: AnswerModel | None = None
    if live_answer:
        settings = get_settings()
        answer_model = OpenAICompatibleAnswerer(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            model=settings.llm_model,
        )
        judge: OfficialBenchmarkJudge | None = OfficialBenchmarkJudge(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
        )
    else:
        judge = None
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
    dataset_sha256 = await asyncio.to_thread(_source_digest, source_path)
    extraction_cache_path = Path(extraction_cache) if extraction_cache else None
    frozen_extractions: dict[str, dict[str, list[dict]]] = {}
    if extraction_cache_path is not None and extraction_cache_path.exists():
        cache_payload = json.loads(extraction_cache_path.read_text())
        expected_metadata = {
            "dataset": dataset,
            "dataset_sha256": dataset_sha256,
            "llm_model": settings.llm_model if live_extraction else None,
        }
        if cache_payload.get("metadata") != expected_metadata:
            raise ValueError("Extraction cache does not match dataset bytes or extraction model")
        frozen_extractions = cache_payload.get("corpora", {})
    protocol_payload = {
        "config_hash": config_hash,
        "dataset_sha256": dataset_sha256,
        "dataset": dataset,
        "system": system_name,
        "limit": limit,
        "live_answer": live_answer,
        "live_extraction": live_extraction,
        "live_embeddings": live_embeddings,
        "storage": storage,
        "ablation": ablation,
        "extraction_policy": "frozen-shared" if extraction_cache_path else "per-run",
        "llm_model": settings.llm_model if live_answer or live_extraction else None,
        "embedding_model": settings.embedding_model if live_embeddings else None,
        "official_judges": live_answer,
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

    def raise_on_failure() -> None:
        failures = [record for record in checkpoint.records.values() if record.failure_type]
        if failures:
            first = failures[0]
            raise RuntimeError(
                f"{dataset}: {len(failures)} of {len(examples)} questions failed; "
                f"first failure: {first.scenario_id}/{first.question_id} "
                f"({first.failure_type}: {first.failure_message}). "
                f"Failure records are preserved in {destination}"
            )

    raise_on_failure()
    if checkpoint.records:
        run_id = next(iter(checkpoint.records.values())).run_id
    client: Neo4jClient | None = None
    if storage == "neo4j":
        if not allow_neo4j_reset:
            raise ValueError(
                "External evaluation clears the selected Neo4j database between corpora; "
                "pass --allow-neo4j-reset only for a disposable evaluation database"
            )
        client = Neo4jClient(settings)
        if not await client.health():
            await client.close()
            raise RuntimeError("Neo4j evaluation database is unavailable")
        await ensure_schema(client)
    elif storage != "memory":
        raise ValueError(f"Unsupported evaluation storage: {storage}")
    active_corpus_id: str | None = None
    repository: InMemoryMemoryRepository | Neo4jMemoryRepository
    system: ScopeGraphMemorySystem
    inputs = []
    memories = []
    corpus_failure: Exception | None = None
    try:
        for example in examples:
            corpus_id = _corpus_id(dataset, example)
            key = (corpus_id, example.question_id)
            if key in checkpoint.records:
                continue
            extraction_usage_before = _total_usage(extractor)
            embedding_usage_before = _total_usage(embedder)
            answer_usage_before = _total_usage(answer_model)
            if corpus_id != active_corpus_id:
                global_scope_id = f"external-global-{corpus_id}"
                scope_id = f"external-{dataset}-{corpus_id}"
                corpus_failure = None
                try:
                    if client is None:
                        repository = InMemoryMemoryRepository()
                    else:
                        await client.execute_write("MATCH (n) DETACH DELETE n")
                        repository = Neo4jMemoryRepository(client)
                    await repository.create_scope(
                        ScopeCreate(
                            id=global_scope_id,
                            name="Global",
                            scope_type=ScopeType.GLOBAL,
                        )
                    )
                    await repository.create_scope(
                        ScopeCreate(
                            id=scope_id,
                            name=f"{dataset}:{corpus_id}",
                            scope_type=ScopeType.CUSTOM,
                            parent_scope_id=global_scope_id,
                        )
                    )
                    if system_name != "scopegraph":
                        raise ValueError(f"Unsupported evaluation system: {system_name}")
                    inputs = session_inputs(
                        example,
                        scope_id=scope_id,
                        as_of=example.question_date,
                        source_namespace=corpus_id,
                    )
                    scope_ref = ScopeRef(
                        id=scope_id,
                        name=f"{dataset}:{corpus_id}",
                        scope_type=ScopeType.CUSTOM,
                    )
                    corpus_extractor: CandidateExtractor = extractor
                    if live_extraction and extraction_cache_path is not None:

                        def persist_external(
                            candidates: dict[str, list[MemoryCandidate]],
                            frozen_corpus_id: str = corpus_id,
                        ) -> None:
                            frozen_extractions[frozen_corpus_id] = {
                                message_id: [
                                    candidate.model_dump(mode="json") for candidate in values
                                ]
                                for message_id, values in candidates.items()
                            }
                            save_json(
                                extraction_cache_path,
                                {
                                    "metadata": {
                                        "dataset": dataset,
                                        "dataset_sha256": dataset_sha256,
                                        "llm_model": settings.llm_model,
                                    },
                                    "corpora": frozen_extractions,
                                },
                            )

                        frozen = await _freeze_external_extraction(
                            inputs,
                            extractor,
                            scope=scope_ref,
                            cached={
                                message_id: [
                                    MemoryCandidate.model_validate(candidate)
                                    for candidate in values
                                ]
                                for message_id, values in frozen_extractions.get(
                                    corpus_id, {}
                                ).items()
                            },
                            checkpoint=persist_external,
                        )
                        persist_external(frozen)
                        corpus_extractor = ScenarioExtractor(frozen)
                    system = ScopeGraphMemorySystem(
                        repository,
                        corpus_extractor,
                        retriever=ScopeAwareRetriever(repository, embedder, retrieval),
                    )
                    for session in inputs:
                        await system.ingest_session(
                            session,
                            current_scope=scope_ref.model_copy(
                                update={"session_id": session.id}
                            ),
                        )
                    memories = await repository.list_memories(include_inactive=True)
                except Exception as exc:  # one failed corpus must not erase the run
                    corpus_failure = exc
                active_corpus_id = corpus_id
            if corpus_failure is not None:
                checkpoint.append(
                    _failure_record(
                        run_id=run_id,
                        dataset=dataset,
                        system_name=system_name,
                        ablation=ablation,
                        corpus_id=corpus_id,
                        example=example,
                        config_hash=config_hash,
                        storage=storage,
                        live_answer=live_answer,
                        live_extraction=live_extraction,
                        live_embeddings=live_embeddings,
                        error=corpus_failure,
                    )
                )
                raise_on_failure()
            gold_sources = evidence_source_ids(
                example,
                inputs,
                source_namespace=corpus_id,
            )
            retrieval_cfg = retrieval_config_data
            top_k = int(retrieval_cfg.get("top_k", 8))
            token_budget = int(retrieval_cfg.get("token_budget", 1500))
            try:
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
                answer = await _answer(answer_model, example, result.items)
                answer_latency = (
                    (time.perf_counter() - answer_started) * 1000 if answer_model else None
                )
                scored = official_score(dataset, answer, example, data_path=source_path)
                judged = (
                    await judge.judge(dataset, example, answer)
                    if judge is not None and answer is not None and scored is None
                    else None
                )
                if judged is not None:
                    scored = (judged.metric, judged.score)
            except Exception as exc:  # preserve failure before stopping the batch
                checkpoint.append(
                    _failure_record(
                        run_id=run_id,
                        dataset=dataset,
                        system_name=system_name,
                        ablation=ablation,
                        corpus_id=corpus_id,
                        example=example,
                        config_hash=config_hash,
                        storage=storage,
                        live_answer=live_answer,
                        live_extraction=live_extraction,
                        live_embeddings=live_embeddings,
                        error=exc,
                    )
                )
                raise_on_failure()
            token_usage = {
                **_usage_delta(
                    extraction_usage_before, _total_usage(extractor), "extraction"
                ),
                **_usage_delta(
                    embedding_usage_before, _total_usage(embedder), "embedding"
                ),
                **_usage_delta(answer_usage_before, _total_usage(answer_model), "answer"),
            }
            stats = await system.stats()
            record = EvaluationRecord(
                protocol_version="external-v4",
                latency_protocol=f"warm-embeddings/{storage}-repository",
                embedding_preparation_ms=preparation_ms,
                run_id=run_id,
                dataset=dataset,
                system=system_name,
                ablation=ablation,
                scenario_id=corpus_id,
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
                retrieved_origin_scope_ids=[[scope_id] for _ in result.items],
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
                benchmark_metadata=example.metadata,
                official_metric=scored[0] if scored else None,
                official_score=scored[1] if scored else None,
                judge_model=judged.model if judged else None,
                judge_latency_ms=judged.latency_ms if judged else None,
                judge_input_tokens=judged.input_tokens if judged else None,
                judge_output_tokens=judged.output_tokens if judged else None,
                token_usage=token_usage,
                evaluation_mode=(
                    f"extraction={'live' if live_extraction else 'turn-preserving'};"
                    f"embeddings={'live' if live_embeddings else 'hash'};"
                    f"answer={'live' if live_answer else 'not-evaluated'};"
                    f"storage={storage}"
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
        await close_provider(judge)
        if embedding_cache is not None:
            embedding_cache.close()
        if client is not None:
            try:
                await client.execute_write("MATCH (n) DETACH DELETE n")
            finally:
                await client.close()
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=EXTERNAL_DATASETS, required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--live-answer", action="store_true")
    parser.add_argument("--live-extraction", action="store_true")
    parser.add_argument("--live-embeddings", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--extraction-cache",
        type=Path,
        help="checkpointed live extraction for this dataset",
    )
    parser.add_argument(
        "--ablation",
        choices=["full"],
        default="full",
    )
    parser.add_argument(
        "--allow-neo4j-reset",
        action="store_true",
        help="allow clearing the configured evaluation-only Neo4j database",
    )
    args = parser.parse_args()
    print(
        asyncio.run(
            run_external(
                dataset=args.dataset,
                path=args.path,
                system_name="scopegraph",
                config_path=args.config,
                output=args.output,
                limit=None,
                live_answer=args.live_answer,
                live_extraction=args.live_extraction,
                live_embeddings=args.live_embeddings,
                live=args.live,
                resume=args.resume,
                storage="neo4j",
                allow_neo4j_reset=args.allow_neo4j_reset,
                ablation=args.ablation,
                extraction_cache=args.extraction_cache,
            )
        )
    )


if __name__ == "__main__":
    main()
