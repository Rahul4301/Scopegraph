"""Command line entry point: python -m memstudy <command>.

Commands that spend money check preflight gates first: tag prereg-v1, and the matching gate in
configs/approvals.yaml. Commands marked (free) never call a model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai

from memstudy.arms.base import Arm
from memstudy.arms.full_context import FullContextArm
from memstudy.arms.mem0_arm import Mem0Arm
from memstudy.arms.rag import RagArm
from memstudy.budget import Budget, ModelPrice, load_prices
from memstudy.coding import DockerSandbox, SweContextBenchGrader, image_for_instance, run_coding
from memstudy.config import DEFAULT_CONFIG, DEFAULT_PRICES, config_hash, load_config
from memstudy.datacheck import verify as verify_data
from memstudy.judge import Judge
from memstudy.llm import ModelCaller
from memstudy.loaders import load_bench
from memstudy.loaders.swectx import load_swectx
from memstudy.metering import CostSink, Meter
from memstudy.phase0 import census
from memstudy.pilot import (
    baseline_check,
    evaluate_judge_check,
    export_judge_check,
    pilot_stats,
    project_full_cost,
    rerun_judge,
    select_pilot,
    stratified_sample,
)
from memstudy.preflight import load_openai_key, require_gates
from memstudy.reader import Reader
from memstudy.runner import finalize_run, run_chat
from memstudy.schema import Item
from memstudy.store import ResultStore

RESULTS = Path("results")
LEDGER = RESULTS / "ledger.jsonl"


def results_root(stage: str) -> Path:
    return RESULTS / ("pilot" if stage == "pilot" else "full")


def stage_gate(stage: str, bench: str) -> str:
    if stage == "pilot":
        return "stage_pilot"
    return f"stage_chat_{bench}" if bench in {"locomo", "longmemeval"} else "stage_coding"


@dataclass
class Runtime:
    cfg: dict[str, Any]
    prices: dict[str, ModelPrice]
    budget: Budget
    client: Any
    reader: Reader
    judge: Judge


def build_runtime(key_var: str = "OPENAI_API_KEY") -> Runtime:
    load_openai_key(key_var=key_var)
    cfg = load_config(DEFAULT_CONFIG)
    prices = load_prices(DEFAULT_PRICES)
    budget = Budget(LEDGER)
    client = openai.OpenAI(max_retries=0)
    reader = Reader(ModelCaller(client, prices[cfg["reader"]["model"]], budget), cfg)
    judge = Judge(ModelCaller(client, prices[cfg["judge"]["model"]], budget), cfg)
    return Runtime(cfg, prices, budget, client, reader, judge)


def build_arm(
    name: str, rt: Runtime, stage: str, tag: dict[str, Any], items: list[Item] | None = None
) -> Arm:
    cfg = rt.cfg
    if name == "A":
        price = rt.prices[cfg["reader"]["model"]]
        assert price.context_window is not None
        counts: dict[str, int] = {}
        for item in items or []:
            counts[item.history_id] = counts.get(item.history_id, 0) + 1
        return FullContextArm(price.context_window, cfg["tokenizer"]["safety_margin"], counts)
    require_gates("g1_extra_models")
    meter = Meter(rt.budget, rt.prices, stage, CostSink(), tag)
    if name == "C":
        return RagArm(cfg["arms"]["C"], meter.wrap(openai.OpenAI(max_retries=0)), meter, Path(".cache/rag"))
    if name == "B":
        return Mem0Arm.create(cfg["arms"]["B"], meter, f".cache/mem0/{stage}")
    raise ValueError(f"unknown arm {name}")


def _write_new_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def cmd_api_check(args: argparse.Namespace) -> None:
    """Two tiny reader calls with the exact request shape the study uses (effort none,
    temperature 0, explicit cache breakpoint on a prefix over 1,024 tokens). Reports what the
    API accepted and returned. Cost is a few hundredths of a cent, recorded in the ledger."""
    if not args.yes:
        raise SystemExit("api-check makes two paid calls (under $0.001); pass --yes to run")
    rt = build_runtime(args.key_var)
    filler = " ".join(f"fact{i} is the number {i * 7 % 101}." for i in range(260))
    results = []
    for _ in range(2):
        call = rt.reader.answer(
            stage="pilot",
            tag={"arm": "check", "item": "api-check"},
            context=filler,
            question="What number is fact7? Reply with the number only.",
            question_date=None,
            cache_prefix=True,
            cache_key="api-check",
        )
        results.append(call)
    report = {
        "model_requested": rt.cfg["reader"]["model"],
        "model_returned": results[0].model_returned,
        "temperature_accepted": True,
        "answers": [r.text for r in results],
        "deterministic_pair": results[0].text == results[1].text,
        "usage": [r.usage.model_dump() for r in results],
        "warnings": [r.warnings for r in results],
        "cost_usd": [r.cost_usd for r in results],
        "ledger_total_usd": rt.budget.spent_total,
    }
    print(json.dumps(report, indent=2))


def cmd_verify_data(args: argparse.Namespace) -> None:
    rows = verify_data()
    print(json.dumps(rows, indent=2))
    if any(r["status"] != "ok" for r in rows):
        raise SystemExit(1)


def cmd_census(args: argparse.Namespace) -> None:
    cfg = load_config()
    price = load_prices(DEFAULT_PRICES)[cfg["reader"]["model"]]
    out: dict[str, Any] = {}
    for bench in ("locomo", "longmemeval"):
        histories, items = load_bench(bench)
        out[bench] = census(histories, items, price, cfg["tokenizer"]["safety_margin"])
        print(
            bench,
            {k: out[bench][k] for k in ("n_histories", "n_items", "tokens", "n_exceed_window")},
            "over long-context threshold:",
            out[bench]["n_over_long_context_threshold"],
        )
    _write_new_json(RESULTS / "phase0" / "token_census.json", out)


def _load_selection(path: Path, items_by_bench: dict[str, list[Item]]) -> dict[str, list[Item]]:
    chosen = json.loads(path.read_text())
    return {
        bench: [i for i in items_by_bench[bench] if i.item_id in set(ids)]
        for bench, ids in chosen.items()
    }


def cmd_pilot_select(args: argparse.Namespace) -> None:
    cfg = load_config()
    items_by_bench = {b: load_bench(b)[1] for b in ("locomo", "longmemeval")}
    selection = select_pilot(items_by_bench, cfg)
    payload = {b: [i.item_id for i in v] for b, v in selection.items()}
    _write_new_json(RESULTS / "pilot" / "selection.json", payload)
    print({b: len(v) for b, v in payload.items()})


def cmd_run(args: argparse.Namespace) -> None:
    require_gates(stage_gate(args.stage, args.bench))
    rt = build_runtime()
    histories, items = load_bench(args.bench)
    if args.stage == "pilot":
        items = _load_selection(RESULTS / "pilot" / "selection.json", {args.bench: items})[args.bench]
    elif args.bench == "locomo" and not args.include_adversarial:
        items = [i for i in items if i.primary]
    run_id = f"{args.stage}-{args.bench}-{args.arm}-{uuid.uuid4().hex[:8]}"
    arm = build_arm(args.arm, rt, args.stage, {"arm": args.arm, "run": run_id}, items)
    store = ResultStore(results_root(args.stage))
    summary = run_chat(
        arm=arm,
        items=items,
        histories=histories,
        reader=rt.reader,
        judge=rt.judge,
        store=store,
        stage=args.stage,
        run_id=run_id,
    )
    run = finalize_run(
        store,
        run_id=run_id,
        arm=args.arm,
        bench=args.bench,
        stage=args.stage,
        cfg_hash=config_hash(rt.cfg),
        summary=summary,
    )
    print(json.dumps({k: run[k] for k in ("run_id", "completed", "errors", "accuracy", "cost_usd")}))


def cmd_judge_export(args: argparse.Namespace) -> None:
    cfg = load_config()
    out = RESULTS / "pilot" / "judge_check"
    info = export_judge_check(
        ResultStore(results_root("pilot")), out, cfg["pilot"]["judge_check_per_arm"], cfg["seed"]
    )
    print(info, "label the human_verdict column of", out / "labeling.csv")


def cmd_judge_rerun(args: argparse.Namespace) -> None:
    require_gates("stage_pilot")
    rt = build_runtime()
    base = RESULTS / "pilot" / "judge_check"
    report = rerun_judge(rt.judge, base / "answer_key.json", base / "rerun.json", "pilot")
    print({k: report[k] for k in ("n", "flips", "flip_rate")})


def cmd_judge_eval(args: argparse.Namespace) -> None:
    cfg = load_config()["pilot"]
    result = evaluate_judge_check(
        Path(args.labeled),
        RESULTS / "pilot" / "judge_check" / "answer_key.json",
        cfg["judge_pass_agreement"],
        cfg["judge_pass_disagreement_gap"],
    )
    print(json.dumps(result, indent=2))


def cmd_coding_baseline(args: argparse.Namespace) -> None:
    cfg = load_config()["pilot"]
    rows = ResultStore(results_root("pilot")).read_items("off", "swectx")
    resolved = sum(1 for r in rows if r["resolved"])
    low, high = cfg["coding_baseline_stop_low"], cfg["coding_baseline_stop_high"]
    print(json.dumps(baseline_check(resolved, len(rows), low, high)))


def cmd_pilot_report(args: argparse.Namespace) -> None:
    cfg = load_config()
    prices = load_prices(DEFAULT_PRICES)
    price = prices[cfg["reader"]["model"]]
    census_path = RESULTS / "phase0" / "token_census.json"
    full_census = json.loads(census_path.read_text())
    store = ResultStore(results_root("pilot"))
    report: dict[str, Any] = {}
    for bench in ("locomo", "longmemeval"):
        histories = {h["history_id"]: h["tokens"] for h in full_census[bench]["histories"]}
        full = [(h["tokens"], h["queries"]) for h in full_census[bench]["histories"]]
        for arm in ("A", "B", "C"):
            if not store.read_items(arm, bench):
                continue
            stats = pilot_stats(store, arm, bench, histories)
            stats["projected_reader_usd"] = project_full_cost(price, arm, stats, full)
            report[f"{arm}/{bench}"] = stats
    _write_new_json(RESULTS / "pilot" / "report" / "pilot_report.json", report)
    print(json.dumps({k: v["projected_reader_usd"] for k, v in report.items()}, indent=2))


def cmd_coding_run(args: argparse.Namespace) -> None:
    require_gates(stage_gate(args.stage, "swectx"))
    rt = build_runtime()
    histories, items = load_swectx(Path(args.data_dir), lite=args.lite)
    if args.stage == "pilot":
        items = stratified_sample(
            items, rt.cfg["pilot"]["coding_tasks"], random.Random(rt.cfg["seed"])
        )
    run_id = f"{args.stage}-swectx-{args.arm}-{uuid.uuid4().hex[:8]}"
    arm = None if args.arm == "off" else build_arm("B", rt, args.stage, {"arm": "B", "run": run_id})
    counts = run_coding(
        items=items,
        histories=histories,
        arm=arm,
        arm_name=args.arm,
        caller=rt.reader.caller,
        grader=SweContextBenchGrader(
            Path(args.bench_repo), args.cases_dir, results_root(args.stage) / "grading"
        ),
        make_sandbox=lambda item: DockerSandbox(image_for_instance(item.item_id)),
        store=ResultStore(results_root(args.stage)),
        stage=args.stage,
        run_id=run_id,
        cfg=rt.cfg,
    )
    print(json.dumps({"run_id": run_id, **counts}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memstudy")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, fn: Any, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(fn=fn)
        return p

    add("verify-data", cmd_verify_data, "(free) check data/ files against the checksum manifest")
    check = add("api-check", cmd_api_check, "(paid, under $0.001) verify the reader request shape")
    check.add_argument("--yes", action="store_true")
    check.add_argument("--key-var", default="OPENAI_API_KEY", help="name of the variable holding the key")
    add("census", cmd_census, "(free) token census of every history")
    add("pilot-select", cmd_pilot_select, "(free) write the deterministic pilot sample")
    run = add("run", cmd_run, "(paid) run one arm on one chat benchmark")
    run.add_argument("--arm", choices=["A", "B", "C"], required=True)
    run.add_argument("--bench", choices=["locomo", "longmemeval"], required=True)
    run.add_argument("--stage", choices=["pilot", "chat"], required=True)
    run.add_argument("--include-adversarial", action="store_true")
    add("judge-export", cmd_judge_export, "(free) export the 100-answer hand-check")
    add("judge-rerun", cmd_judge_rerun, "(paid, tiny) rerun nano on the hand-check items")
    ev = add("judge-eval", cmd_judge_eval, "(free) score the hand-labeled CSV against nano")
    ev.add_argument("labeled")
    add("coding-baseline", cmd_coding_baseline, "(free) memory-off solve rate gate")
    add("pilot-report", cmd_pilot_report, "(free) per-arm tokens, cost, latency, projection")
    code = add("coding-run", cmd_coding_run, "(paid) run the coding benchmark, memory off or Mem0")
    code.add_argument("--arm", choices=["off", "B"], required=True)
    code.add_argument("--stage", choices=["pilot", "coding"], required=True)
    code.add_argument("--data-dir", default="data/swectx/data")
    code.add_argument("--lite", action="store_true")
    code.add_argument("--bench-repo", default="third_party/SWEContextBench", help="pinned benchmark clone")
    code.add_argument("--cases-dir", default="cases/SWEContextBench Lite", help="relative to --bench-repo")
    args = parser.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
