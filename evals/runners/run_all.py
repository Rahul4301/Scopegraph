"""Run CrossScopeMem across all controlled memory systems."""

import argparse
import asyncio
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from evals.adapters.cross_scope_mem import CrossScopeMemAdapter
from evals.runners.providers import (
    EvaluationProviders,
    build_cross_scope_providers,
    freeze_extraction,
)
from evals.runners.run_eval import run_evaluation
from scopegraph.config import get_settings


async def run_all(*, systems: list[str], **kwargs: object) -> list[str]:
    if not systems or len(systems) != len(set(systems)):
        raise ValueError("Supply distinct evaluation systems")
    if set(systems) - {"vector_memory", "flat_graph", "two_level_graph", "scopegraph"}:
        raise ValueError("Unknown evaluation system")
    batch = Path(str(kwargs.pop("output", "results/batches"))) / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    batch.mkdir(parents=True, exist_ok=False)
    scenarios = CrossScopeMemAdapter(
        seed=int(str(kwargs.get("seed", 42))),
        difficulty=int(str(kwargs.get("difficulty", 2))),
        scenario_count=int(str(kwargs.get("scenario_count", 1))),
    ).scenarios()
    bank: dict[str, EvaluationProviders] = {}
    settings = get_settings()
    diff_result = await asyncio.to_thread(
        subprocess.run, ["git", "diff", "HEAD"], capture_output=True, check=False
    )
    diff = diff_result.stdout
    manifest = {
        "protocol": "cross-scope-v2", "status": "preparing", "systems": systems,
        "options": kwargs, "paths": [],
        "llm_model": settings.llm_model, "embedding_model": settings.embedding_model,
        "working_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "extraction_policy": "frozen once per source; no backend-specific existing memories",
        "latency_protocol": "warm-embeddings/in-memory-repository",
        "limitations": ["synthetic fixture; seeds shuffle order, not independent worlds",
                        "offline runs evaluate retrieval, not model answer quality",
                        "retrieval latency excludes extraction and embedding API preparation"],
    }
    manifest_path = batch / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    paths = []
    try:
        frozen = {}
        for scenario in scenarios:
            providers = build_cross_scope_providers(
                scenario,
                live_extraction=bool(kwargs.get("live") or kwargs.get("live_extraction")),
                live_embeddings=bool(kwargs.get("live") or kwargs.get("live_embeddings")),
            )
            bank[scenario.scenario_id] = providers
            candidates = await freeze_extraction(scenario, providers)
            frozen[scenario.scenario_id] = {
                key: [candidate.model_dump(mode="json") for candidate in values]
                for key, values in candidates.items()
            }
        artifact = json.dumps(frozen, indent=2, sort_keys=True)
        (batch / "extractions.json").write_text(artifact + "\n")
        manifest["extractions_sha256"] = hashlib.sha256(artifact.encode()).hexdigest()
        for system in systems:
            paths.append(str(await run_evaluation(
                system_name=system, output=str(batch / f"{system}.jsonl"),
                provider_bank=bank, **kwargs,
            )))
        manifest["status"] = "complete"
        return paths
    except Exception:
        manifest["status"] = "failed"
        raise
    finally:
        manifest["paths"] = paths
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        for providers in bank.values():
            providers.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cross_scope_mem")
    parser.add_argument("--systems", default="vector_memory,flat_graph,two_level_graph,scopegraph")
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--scenario-count", type=int, default=1)
    parser.add_argument("--live-answer", action="store_true")
    parser.add_argument("--live-extraction", action="store_true")
    parser.add_argument("--live-embeddings", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.dataset != "cross_scope_mem":
        raise SystemExit("run_all supports cross_scope_mem; use run_external for external datasets")
    paths = asyncio.run(run_all(systems=[item.strip() for item in args.systems.split(",")],
                                seed=args.seed, difficulty=args.difficulty,
                                scenario_count=args.scenario_count, config_path=args.config,
                                live_answer=args.live_answer, live_extraction=args.live_extraction,
                                live_embeddings=args.live_embeddings, live=args.live))
    print("\n".join(paths))


if __name__ == "__main__":
    main()
