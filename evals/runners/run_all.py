"""Run CrossScopeMem across all controlled memory systems."""

import argparse
import asyncio

from evals.runners.run_eval import run_evaluation


async def run_all(*, systems: list[str], **kwargs: object) -> list[str]:
    paths = []
    for system in systems:
        paths.append(str(await run_evaluation(system_name=system, **kwargs)))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cross_scope_mem")
    parser.add_argument("--systems", default="vector_memory,flat_graph,two_level_graph,scopegraph")
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--scenario-count", type=int, default=1)
    args = parser.parse_args()
    if args.dataset != "cross_scope_mem":
        raise SystemExit("run_all supports cross_scope_mem; use run_external for external datasets")
    paths = asyncio.run(run_all(systems=[item.strip() for item in args.systems.split(",")],
                                seed=args.seed, difficulty=args.difficulty,
                                scenario_count=args.scenario_count, config_path=args.config))
    print("\n".join(paths))


if __name__ == "__main__":
    main()
