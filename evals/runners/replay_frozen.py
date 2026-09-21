"""Re-evaluate a saved batch using its extractions and cached vectors, with no API calls."""

import argparse
import asyncio
import json
from pathlib import Path

from evals.adapters.cross_scope_mem import ScenarioExtractor
from evals.analysis.aggregate import aggregate_records
from evals.runners.checkpoint import save_json, source_fingerprint
from evals.runners.providers import EvaluationProviders, PreparedEmbedder
from evals.runners.run_eval import run_scenario
from evals.scenarios.generate_cross_scope import generate_cross_scope_mem
from evals.schemas import CrossScopeScenario
from scopegraph.config import get_settings
from scopegraph.embeddings.cache import SQLiteEmbeddingCache, content_hash
from scopegraph.models.memory import MemoryCandidate


class CachedOnlyEmbedder:
    def __init__(self, model: str, cache: SQLiteEmbeddingCache):
        self.model_name = model
        self.cache = cache

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = [self.cache.get(content_hash(self.model_name, text)) for text in texts]
        if any(vector is None for vector in vectors):
            raise ValueError(
                "Required cached embedding is missing; frozen replay makes no API calls"
            )
        return [vector for vector in vectors if vector is not None]


async def replay(batch: Path, output: Path) -> None:
    manifest = json.loads((batch / "manifest.json").read_text())
    frozen = json.loads((batch / "extractions.json").read_text())
    scenario_path = batch / "scenarios.json"
    if scenario_path.exists():
        scenarios = [CrossScopeScenario.model_validate(item)
                     for item in json.loads(scenario_path.read_text())]
    elif manifest["protocol"] == "cross-scope-v2":
        options = manifest["options"]
        scenarios = [generate_cross_scope_mem(
            seed=options["seed"] + index, difficulty=options["difficulty"],
            scenario_id=f"cross_scope_mem_{options['seed'] + index}_{options['difficulty']}",
        ) for index in range(options["scenario_count"])]
    else:
        raise ValueError("Batch has no recorded scenarios")
    cache = SQLiteEmbeddingCache(get_settings().embedding_cache_path)
    rows = []
    try:
        for scenario in scenarios:
            providers = EvaluationProviders(ScenarioExtractor({
                key: [MemoryCandidate.model_validate(candidate) for candidate in values]
                for key, values in frozen[scenario.scenario_id].items()
            }), PreparedEmbedder(CachedOnlyEmbedder(manifest["embedding_model"], cache)))
            for system in manifest["systems"]:
                rows.extend(await run_scenario(
                    scenario, system_name=system, run_id=f"replay_{batch.name}",
                    config={"evaluation_mode": "frozen-live-extraction/cached-vectors/no-answers"},
                    config_hash=source_fingerprint(), providers=providers,
                ))
    finally:
        cache.close()
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(
        (output / "replay.jsonl").write_text,
        "".join(row.model_dump_json() + "\n" for row in rows),
    )
    summary = aggregate_records(rows)
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/audit-report/frozen-replay"))
    args = parser.parse_args()
    asyncio.run(replay(args.batch, args.output))


if __name__ == "__main__":
    main()
