"""Run one of the controlled ScopeGraph retrieval ablations."""

import argparse
import asyncio

from evals.runners.run_eval import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cross_scope_mem")
    parser.add_argument("--system", default="scopegraph")
    parser.add_argument("--ablation", required=True,
                        choices=["full", "no_scope_weighting", "no_graph_traversal"])
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=int, default=2)
    args = parser.parse_args()
    if args.dataset != "cross_scope_mem":
        raise SystemExit("run_ablation supports cross_scope_mem only")
    print(asyncio.run(run_evaluation(system_name=args.system, seed=args.seed,
                                     difficulty=args.difficulty, config_path=args.config,
                                     ablation=args.ablation)))


if __name__ == "__main__":
    main()
