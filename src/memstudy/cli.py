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
from memstudy.asof import expand_asof
from memstudy.budget import Budget, ModelPrice, load_prices, worst_case_usd
from memstudy.config import DEFAULT_CONFIG, DEFAULT_PRICES, config_hash, load_config
from memstudy.datacheck import verify as verify_data
from memstudy.execute import (
    EVALS,
    SYSTEMS,
    LivePrinter,
    ensure_new,
    execute_run,
    format_summary,
    resolve_system,
    result_name,
    result_path,
    select_cases,
)
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
from memstudy.schema import Item, render_transcript
from memstudy.store import ResultStore
from memstudy.tokens import check_fit, count_tokens

RESULTS = Path("results")
LEDGER = RESULTS / "ledger.jsonl"


def results_root(stage: str) -> Path:
    return RESULTS / ("pilot" if stage == "pilot" else "full")


def stage_gate(stage: str, bench: str) -> str:
    if stage in ("pilot", "smoke"):
        return f"stage_{stage}"
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
    name: str,
    rt: Runtime,
    stage: str,
    tag: dict[str, Any],
    items: list[Item] | None = None,
    store_name: str | None = None,
) -> Arm:
    """store_name keys the Mem0 store directory so separate runs never share (and duplicate)
    memories; it defaults to the stage."""
    cfg = rt.cfg
    if name == "A":
        price = rt.prices[cfg["reader"]["model"]]
        assert price.context_window is not None
        # Questions per memory store. As-of checkpoints of one conversation share a store, and
        # each later checkpoint extends the previous prefix, so every checkpoint history counts
        # all of its conversation's questions and gets the cache breakpoint.
        per_store: dict[str, int] = {}
        for item in items or []:
            key = str(item.meta.get("store_id", item.history_id))
            per_store[key] = per_store.get(key, 0) + 1
        counts = {
            item.history_id: per_store[str(item.meta.get("store_id", item.history_id))]
            for item in items or []
        }
        return FullContextArm(price.context_window, cfg["tokenizer"]["safety_margin"], counts)
    meter = Meter(rt.budget, rt.prices, stage, CostSink(), tag)
    retrieval = cfg["retrieval"]
    require_gates("g1_extra_models")
    if name == "C":
        embedder = meter.wrap(openai.OpenAI(max_retries=0))
        return RagArm(cfg["arms"]["C"], retrieval, embedder, meter, Path(".cache/rag"))
    if name == "B":
        return Mem0Arm.create(cfg["arms"]["B"], retrieval, meter, f".cache/mem0/{store_name or stage}")
    if name == "D":
        arm_cfg = cfg["arms"]["D"]
        require_gates("g2_supermemory_self_hosted")
        load_env_key("SUPERMEMORY_API_KEY")  # the key the local server printed on first boot
        from supermemory import Supermemory

        return SupermemoryArm(
            arm_cfg,
            retrieval,
            Supermemory(base_url=arm_cfg["base_url"], max_retries=0),
            rt.prices[arm_cfg["extraction_model"]],
            rt.prices[arm_cfg["embedding_model"]],
            rt.budget,
            stage,
            tag,
        )
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


def _dry_run(
    args: argparse.Namespace,
    stage: str,
    name: str,
    letter: str,
    cases: list[Item],
    histories: dict[str, Any],
) -> None:
    """Free preview of a run: which cases, how long their histories are, and the reader's worst
    case cost. No model call, no gate, no key."""
    cfg = load_config()
    price = load_prices(DEFAULT_PRICES)[cfg["reader"]["model"]]
    assert price.context_window is not None
    margin = cfg["tokenizer"]["safety_margin"]
    tokens: dict[str, int] = {}
    rows: dict[str, dict[str, Any]] = {}
    for item in cases:
        if item.history_id not in tokens:
            tokens[item.history_id] = count_tokens(render_transcript(histories[item.history_id]))
        full = tokens[item.history_id]
        fits = check_fit(full + 600, price.context_window, margin, price.long_context_threshold).fits
        prompt = full + 600 if letter == "A" else cfg["retrieval"]["token_budget"] + 600
        row = rows.setdefault(
            item.history_id,
            {
                "history_id": item.history_id,
                "questions": 0,
                "history_tokens": full,
                "fits_window": fits,
                "reader_worst_case_usd": 0.0,
            },
        )
        row["questions"] += 1
        row["reader_worst_case_usd"] += worst_case_usd(price, prompt, cfg["reader"]["max_output_tokens"])
    gates = [stage_gate(stage, args.eval)]
    if letter in ("B", "C", "D"):
        gates.append("g1_extra_models")
    if letter == "D":
        gates.append("g2_supermemory_self_hosted")
    print(
        json.dumps(
            {
                "dry_run": True,
                "would_write": str(result_path(RESULTS, name)),
                "stage": stage,
                "gates_needed": ["tag prereg-v1", *gates],
                "cases": len(rows),
                "questions": len(cases),
                "history_tokens_total": sum(tokens.values()),
                "reader_worst_case_usd_total": sum(r["reader_worst_case_usd"] for r in rows.values()),
                "case_list": list(rows.values()),
            },
            indent=2,
        )
    )


def _run_pilot(args: argparse.Namespace, letter: str) -> None:
    """Legacy pilot: the fixed selection from `pilot-select`, raw records under results/pilot."""
    require_gates(stage_gate("pilot", args.eval))
    rt = build_runtime(args.key_var)
    histories, items = load_bench(args.eval)
    items = _load_selection(RESULTS / "pilot" / "selection.json", {args.eval: items})[args.eval]
    run_id = f"pilot-{args.eval}-{letter}-{uuid.uuid4().hex[:8]}"
    arm = build_arm(letter, rt, "pilot", {"arm": letter, "run": run_id}, items)
    store = ResultStore(results_root("pilot"))
    summary = run_chat(
        arm=arm, items=items, histories=histories, reader=rt.reader, judge=rt.judge,
        store=store, stage="pilot", run_id=run_id,
    )
    run = finalize_run(
        store, run_id=run_id, arm=letter, bench=args.eval, stage="pilot",
        cfg_hash=config_hash(rt.cfg), summary=summary,
    )
    print(json.dumps({k: run[k] for k in ("run_id", "completed", "errors", "accuracy", "cost_usd")}))


def cmd_run(args: argparse.Namespace) -> None:
    system, letter = resolve_system(args.memory_system)
    stage = args.stage or ("smoke" if args.num_cases is not None else "chat")
    if stage == "pilot":
        _run_pilot(args, letter)
        return
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.asof and args.eval != "locomo":
        raise SystemExit("--asof applies to locomo only")
    if args.write_granularity and letter != "B":
        raise SystemExit("--write_granularity applies to the Mem0 arm (--memory_system mem0) only")
    label = args.eval + ("_asof" if args.asof else "")
    if args.write_granularity and args.write_granularity != "ten":
        label += f"_{args.write_granularity}"
    name = result_name(system, label)
    ensure_new(result_path(RESULTS, name))
    if not args.dry_run:
        require_gates(stage_gate(stage, args.eval))
    histories, items = load_bench(args.eval)
    cases = select_cases(items, args.num_cases, args.include_secondary)
    if not cases:
        raise SystemExit(f"no cases selected from {args.eval}")
    if args.asof:
        histories, cases = expand_asof(histories, cases)
    if args.dry_run:
        _dry_run(args, stage, name, letter, cases, histories)
        return
    rt = build_runtime(args.key_var)
    if args.write_granularity:
        rt.cfg["arms"]["B"]["write_granularity"] = args.write_granularity  # part of the config hash
    # A smoke run counts against the pilot budget cap; the pre-registered caps are unchanged.
    budget_stage = "pilot" if stage == "smoke" else stage
    arm = build_arm(letter, rt, budget_stage, {"arm": letter, "run": name}, cases, store_name=name)
    path = execute_run(
        system=system,
        eval_name=args.eval,
        num_cases=args.num_cases,
        arm=arm,
        cases=cases,
        histories=histories,
        reader=rt.reader,
        judge=rt.judge,
        stage=stage,
        budget_stage=budget_stage,
        results_dir=RESULTS,
        cfg=rt.cfg,
        cfg_hash=config_hash(rt.cfg),
        name=name,
        progress=LivePrinter(len(cases)),
        workers=args.workers,
    )
    print(format_summary(json.loads(path.read_text())))
    print(f"\nresults: {path}")


def cmd_judge_flip(args: argparse.Namespace) -> None:
    """Rerun the judge on 100 answers (50 arm A, 50 arm B) and report its flip rate."""
    require_gates("stage_pilot")
    rt = build_runtime()
    store = ResultStore(results_root("pilot"))
    sample = select_flip_sample(store, rt.cfg["pilot"]["judge_flip_per_arm"], rt.cfg["seed"])
    report = rerun_judge(rt.judge, sample, RESULTS / "pilot" / "judge_flip.json", "pilot")
    print({k: report[k] for k in ("n", "flips", "flip_rate", "flip_rate_by_arm")})


def cmd_judge_diagnostics(args: argparse.Namespace) -> None:
    """Judge agreement with a deterministic containment test on short answers (free). With
    --file, reads one run result file (results/<system>_<eval>_<time>.json) instead."""
    if args.file:
        report = json.loads(Path(args.file).read_text())
        records = [
            {
                "arm": report["run"]["arm"],
                "item_id": c["item_id"],
                "gold": c["gold"][0],
                "model_answer": c["answer"],
                "correct": c["correct"],
                "abstention": False,
            }
            for c in report["cases"]
            if c["status"] == "ok"
        ]
    else:
        roots = (
            [results_root("pilot")] if args.stage == "pilot" else sorted((RESULTS / "runs").glob("*"))
        )
        records = [
            r
            for root in roots
            for arm in ("A", "B", "C")
            for b in EVALS
            for r in ResultStore(root).read_items(arm, b)
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
        for arm in ("A", "B", "C"):
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
    run = add("run", cmd_run, "(paid) run one memory system on one eval and write its result file")
    run.add_argument(
        "--memory_system",
        required=True,
        choices=[*SYSTEMS, *SYSTEMS.values()],
        metavar="SYSTEM",
        help="no_memory (A), mem0 (B), rag (C) or supermemory (D, self-hosted, exploratory)",
    )
    run.add_argument("--eval", required=True, choices=EVALS, help="benchmark to run")
    run.add_argument(
        "--num_cases",
        type=int,
        default=None,
        help="smoke test on the first N cases (a case is one conversation/history with all of its "
        "questions); omit for the full run. Result file: "
        "results/<system>_<eval>_<YYYYMMDD_HHMMSS>.json",
    )
    run.add_argument(
        "--include_secondary",
        action="store_true",
        help="also run non-primary cases (LoCoMo adversarial, MemoryAgentBench exploratory)",
    )
    run.add_argument(
        "--workers",
        type=int,
        default=4,
        help="questions answered at the same time (default 4; 1 for strictly one at a time)",
    )
    run.add_argument(
        "--asof",
        action="store_true",
        help="locomo only: ask each question after the session holding its last evidence turn "
        "and again at the end (exploratory, PREREG.md Section 10)",
    )
    run.add_argument(
        "--write_granularity",
        choices=["turn", "ten", "session"],
        help="mem0 only: messages per ingestion call; the registered default is ten",
    )
    run.add_argument("--dry_run", action="store_true", help="(free) preview cases and worst-case cost")
    run.add_argument("--key_var", default="OPENAI_API_KEY", help="name of the variable holding the OpenAI key")
    run.add_argument("--stage", choices=["smoke", "pilot", "chat"], help=argparse.SUPPRESS)
    add("judge-flip", cmd_judge_flip, "(paid, tiny) rerun nano on 100 answers for its flip rate")
    diag = add("judge-diagnostics", cmd_judge_diagnostics, "(free) judge vs containment test")
    diag.add_argument("--file", help="a run result file to check instead of the stored records")
    diag.add_argument(
        "--stage",
        choices=["pilot", "runs"],
        default="pilot",
        help="pilot, or runs for every run under results/runs",
    )
    add("pilot-report", cmd_pilot_report, "(free) per-arm tokens, cost, latency, projection")
    args = parser.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
