"""Command line entry point: python -m memstudy <command>.

Commands that spend money check preflight gates first: tag prereg-v1, and the matching gate in
configs/approvals.yaml. Commands marked (free) never call a model.
"""

from __future__ import annotations

import argparse
import json
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
from memstudy.arms.supermemory_arm import SupermemoryArm
from memstudy.budget import Budget, ModelPrice, load_prices, load_supermemory_price
from memstudy.config import DEFAULT_CONFIG, DEFAULT_PRICES, config_hash, load_config
from memstudy.datacheck import verify as verify_data
from memstudy.judge import Judge
from memstudy.llm import ModelCaller
from memstudy.loaders import load_bench
from memstudy.metering import CostSink, Meter
from memstudy.phase0 import census
from memstudy.pilot import (
    containment_diagnostic,
    pilot_stats,
    project_full_cost,
    rerun_judge,
    select_flip_sample,
    select_pilot,
)
from memstudy.preflight import load_env_key, load_openai_key, require_gates
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
    return f"stage_chat_{bench}"


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
    meter = Meter(rt.budget, rt.prices, stage, CostSink(), tag)
    retrieval = cfg["retrieval"]
    if name == "C":
        require_gates("g2_supermemory")
        load_env_key("SUPERMEMORY_API_KEY")
        from supermemory import Supermemory

        return SupermemoryArm(
            cfg["arms"]["C"],
            retrieval,
            Supermemory(),
            load_supermemory_price(DEFAULT_PRICES),
            rt.budget,
            stage,
            tag,
        )
    require_gates("g1_extra_models")
    if name == "D":
        embedder = meter.wrap(openai.OpenAI(max_retries=0))
        return RagArm(cfg["arms"]["D"], retrieval, embedder, meter, Path(".cache/rag"))
    if name == "B":
        return Mem0Arm.create(cfg["arms"]["B"], retrieval, meter, f".cache/mem0/{stage}")
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


def cmd_judge_flip(args: argparse.Namespace) -> None:
    """Rerun the judge on 100 answers (50 arm A, 50 arm B) and report its flip rate."""
    require_gates("stage_pilot")
    rt = build_runtime()
    store = ResultStore(results_root("pilot"))
    sample = select_flip_sample(store, rt.cfg["pilot"]["judge_flip_per_arm"], rt.cfg["seed"])
    report = rerun_judge(rt.judge, sample, RESULTS / "pilot" / "judge_flip.json", "pilot")
    print({k: report[k] for k in ("n", "flips", "flip_rate", "flip_rate_by_arm")})


def cmd_judge_diagnostics(args: argparse.Namespace) -> None:
    """Judge agreement with a deterministic containment test on short answers (free)."""
    store = ResultStore(results_root(args.stage))
    records = [
        r for arm in ("A", "B", "C", "D") for b in ("locomo", "longmemeval") for r in store.read_items(arm, b)
    ]
    print(json.dumps(containment_diagnostic(records), indent=2))


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
        for arm in ("A", "B", "C", "D"):
            if not store.read_items(arm, bench):
                continue
            stats = pilot_stats(store, arm, bench, histories)
            stats["projected_reader_usd"] = project_full_cost(price, arm, stats, full)
            report[f"{arm}/{bench}"] = stats
    _write_new_json(RESULTS / "pilot" / "report" / "pilot_report.json", report)
    print(json.dumps({k: v["projected_reader_usd"] for k, v in report.items()}, indent=2))


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
    run.add_argument("--arm", choices=["A", "B", "C", "D"], required=True)
    run.add_argument("--bench", choices=["locomo", "longmemeval"], required=True)
    run.add_argument("--stage", choices=["pilot", "chat"], required=True)
    run.add_argument("--include-adversarial", action="store_true")
    add("judge-flip", cmd_judge_flip, "(paid, tiny) rerun nano on 100 answers for its flip rate")
    diag = add("judge-diagnostics", cmd_judge_diagnostics, "(free) judge vs containment test")
    diag.add_argument("--stage", choices=["pilot", "chat"], default="pilot")
    add("pilot-report", cmd_pilot_report, "(free) per-arm tokens, cost, latency, projection")
    args = parser.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
