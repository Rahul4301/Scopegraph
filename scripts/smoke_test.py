"""Run the credential-free evaluation smoke test and verify its JSONL output."""

import argparse
import asyncio
import json
from pathlib import Path

from evals.runners.run_all import run_all


async def _run() -> list[Path]:
    paths = await run_all(
        seed=42,
        difficulty=1,
        scenario_count=1,
        config_path="configs/experiments.yaml",
    )
    checked: list[Path] = []
    for raw_path in paths:
        source = Path(raw_path)
        checked.append(source)
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("results/smoke"))
    args = parser.parse_args()
    paths = asyncio.run(_run())
    for source in paths:
        records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if not records:
            raise RuntimeError(f"Smoke run produced no records: {source}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "manifest.json"
    manifest.write_text(json.dumps({"paths": [str(path) for path in paths]}, indent=2) + "\n")
    print("Smoke test passed for ScopeGraph.")
    print(args.output_root / "manifest.json")


if __name__ == "__main__":
    main()
