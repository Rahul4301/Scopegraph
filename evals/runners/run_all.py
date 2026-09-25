"""Run a reproducible ScopeGraph CrossScopeMem batch."""

import argparse
import asyncio
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from evals.adapters.cross_scope_mem import CrossScopeMemAdapter
from evals.analysis.scope_classification import evaluate_scope_classification
from evals.runners.checkpoint import save_json, source_fingerprint
from evals.runners.providers import (
    EvaluationProviders,
    build_cross_scope_providers,
    freeze_extraction,
)
from evals.runners.run_eval import run_evaluation
from evals.schemas import EvaluationRecord
from scopegraph.config import get_settings
from scopegraph.models.memory import MemoryCandidate

ABLATIONS = (
    "full",
    "vector_only_control",
    "flat_graph_control",
    "two_level_control",
    "no_graph_traversal",
    "no_temporal_status",
)


async def run_all(**kwargs: object) -> list[str]:
    resume_path = kwargs.pop("resume", None)
    output_root = kwargs.pop("output", "results/batches")
    batch = (
        Path(str(resume_path))
        if resume_path
        else Path(str(output_root)) / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    if not resume_path:
        batch.mkdir(parents=True, exist_ok=False)
    scenarios = CrossScopeMemAdapter(
        seed=int(str(kwargs.get("seed", 42))),
        difficulty=int(str(kwargs.get("difficulty", 2))),
        scenario_count=int(str(kwargs.get("scenario_count", 1))),
        profile=str(kwargs.get("profile", "research")),
    ).scenarios()
    bank: dict[str, EvaluationProviders] = {}
    settings = get_settings()
    diff_result = await asyncio.to_thread(
        subprocess.run, ["git", "diff", "HEAD"], capture_output=True, check=False
    )
    diff = diff_result.stdout
    manifest = {
        "protocol": f"cross-scope-v4/{kwargs.get('profile', 'research')}",
        "status": "preparing",
        "system": "scopegraph",
        "options": kwargs,
        "paths": [],
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "working_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_fingerprint": source_fingerprint(),
        "extraction_policy": "frozen once per source; no backend-specific existing memories",
        "ablations": list(ABLATIONS),
        "latency_protocol": (
            f"warm-embeddings/{kwargs.get('storage', 'memory')}-repository"
        ),
        "limitations": [
            "synthetic templates; use external datasets before generalizing",
            "offline runs evaluate retrieval, not model answer quality",
            "retrieval latency excludes extraction and embedding API preparation",
        ],
    }
    manifest_path = batch / "manifest.json"
    if resume_path:
        previous = json.loads(manifest_path.read_text())
        for key in ("options", "system", "source_fingerprint", "llm_model", "embedding_model"):
            if previous.get(key) != manifest[key]:
                raise ValueError(f"Cannot resume: {key} changed from the original batch")
    save_json(manifest_path, manifest)
    save_json(
        batch / "scenarios.json", [scenario.model_dump(mode="json") for scenario in scenarios]
    )
    paths = []
    try:
        extraction_path = batch / "extractions.json"
        frozen = (
            json.loads(extraction_path.read_text())
            if resume_path and extraction_path.exists()
            else {}
        )
        for scenario in scenarios:
            providers = build_cross_scope_providers(
                scenario,
                live_extraction=bool(kwargs.get("live") or kwargs.get("live_extraction")),
                live_embeddings=bool(kwargs.get("live") or kwargs.get("live_embeddings")),
            )
            bank[scenario.scenario_id] = providers

            def persist(candidates, scenario_id=scenario.scenario_id):
                frozen[scenario_id] = {
                    key: [candidate.model_dump(mode="json") for candidate in values]
                    for key, values in candidates.items()
                }
                save_json(extraction_path, frozen)

            candidates = await freeze_extraction(
                scenario,
                providers,
                cached={
                    key: [MemoryCandidate.model_validate(candidate) for candidate in values]
                    for key, values in frozen.get(scenario.scenario_id, {}).items()
                },
                checkpoint=persist,
            )
            frozen[scenario.scenario_id] = {
                key: [candidate.model_dump(mode="json") for candidate in values]
                for key, values in candidates.items()
            }
        artifact = json.dumps(frozen, indent=2, sort_keys=True)
        save_json(extraction_path, frozen)
        manifest["extractions_sha256"] = hashlib.sha256(artifact.encode()).hexdigest()
        classification = evaluate_scope_classification(
            scenarios,
            {
                scenario_id: {
                    message_id: [MemoryCandidate.model_validate(candidate) for candidate in values]
                    for message_id, values in by_message.items()
                }
                for scenario_id, by_message in frozen.items()
            },
            live_extraction=bool(kwargs.get("live") or kwargs.get("live_extraction")),
        )
        save_json(batch / "classification.json", classification.model_dump(mode="json"))
        manifest["classification_evaluated"] = classification.evaluated
        total_questions = sum(len(scenario.examples) for scenario in scenarios) * len(ABLATIONS)
        completed = 0

        def report(record: EvaluationRecord) -> None:
            nonlocal completed
            completed += 1
            print(
                f"[scopegraph/{record.ablation}] {completed}/{total_questions} | "
                f"{record.scenario_id} | {record.question_id} | "
                f"retrieval={record.retrieval_latency_ms:.1f}ms",
                flush=True,
            )

        for ablation in ABLATIONS:
            paths.append(
                str(
                    await run_evaluation(
                        system_name="scopegraph",
                        output=str(batch / f"scopegraph_{ablation}.jsonl"),
                        provider_bank=bank,
                        resume=bool(resume_path),
                        on_progress=report,
                        ablation=ablation,
                        **kwargs,
                    )
                )
            )
        manifest["status"] = "complete"
        return paths
    except Exception:
        manifest["status"] = "failed"
        raise
    finally:
        manifest["paths"] = paths
        save_json(manifest_path, manifest)
        for providers in bank.values():
            await providers.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cross_scope_mem")
    parser.add_argument("--config")
    parser.add_argument("--output", default="results/batches")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--scenario-count", type=int, default=1)
    parser.add_argument("--live-answer", action="store_true")
    parser.add_argument("--live-extraction", action="store_true")
    parser.add_argument("--live-embeddings", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--profile", choices=["smoke", "research"], default="research")
    parser.add_argument(
        "--resume", help="Resume an existing batch directory with identical options"
    )
    parser.add_argument(
        "--allow-neo4j-reset",
        action="store_true",
        help="allow clearing the configured evaluation-only Neo4j database",
    )
    args = parser.parse_args()
    if args.dataset != "cross_scope_mem":
        raise SystemExit("run_all supports cross_scope_mem; use run_external for external datasets")
    paths = asyncio.run(
        run_all(
            seed=args.seed,
            difficulty=args.difficulty,
            scenario_count=args.scenario_count,
            config_path=args.config,
            output=args.output,
            live_answer=args.live_answer,
            live_extraction=args.live_extraction,
            live_embeddings=args.live_embeddings,
            live=args.live,
            resume=args.resume,
            profile=args.profile,
            storage="neo4j",
            allow_neo4j_reset=args.allow_neo4j_reset,
        )
    )
    print("\n".join(paths))


if __name__ == "__main__":
    main()
