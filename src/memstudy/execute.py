"""One named run: pick cases, run an arm over them, write one readable result file.

    python -m memstudy run --memory_system mem0 --eval locomo --num_cases 1

writes results/mem0_locomo_1.json (results/mem0_locomo.json for a full run). The file is created
exclusively and never overwritten. The per-item raw records it is built from live under
results/runs/<name>/ and are what a resumed run reads, so an interrupted run loses no paid work.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memstudy.arms.base import Arm
from memstudy.judge import Judge
from memstudy.reader import Reader
from memstudy.runner import RunSummary, finalize_run, run_chat, summarize_records, versions
from memstudy.schema import History, Item
from memstudy.store import ResultStore

SYSTEMS = {"no_memory": "A", "mem0": "B", "rag": "C"}
EVALS = ("locomo", "longmemeval", "memoryagentbench")

METRIC_NOTES = {
    "accuracy.judge": "Share of cases GPT-5 nano graded CORRECT (the pre-registered primary metric).",
    "accuracy.exact_match": "Normalized answer equals a normalized accepted answer.",
    "accuracy.substring_match": "A normalized accepted answer appears inside the normalized answer.",
    "accuracy.token_f1": "Mean best token-overlap F1 against the accepted answers.",
    "accuracy.gold_in_context": "Share of cases where an accepted answer appears verbatim in the "
    "context the reader saw (retrieval proxy; short answers can match by chance).",
    "scored_cases": "Abstention cases have no answer string, so only the judge scores them and "
    "they are left out of exact_match, substring_match, token_f1 and gold_in_context.",
    "cost_usd": "Reader, judge, retrieval and ingestion spend, from the provider's reported "
    "usage and the verified prices in configs/prices.yaml.",
    "latency_ms": "Per-case wall time; ingestion time is under ingest.seconds.",
}


class ResultExists(RuntimeError):
    pass


def resolve_system(value: str) -> tuple[str, str]:
    """(system name, arm letter) from either spelling."""
    if value in SYSTEMS:
        return value, SYSTEMS[value]
    for name, letter in SYSTEMS.items():
        if value == letter:
            return name, letter
    raise ValueError(f"unknown memory system {value!r}; use one of {', '.join(SYSTEMS)}")


def result_name(system: str, eval_name: str, num_cases: int | None) -> str:
    """system_eval_numcases, or system_eval for a full run."""
    base = f"{system}_{eval_name}"
    return base if num_cases is None else f"{base}_{num_cases}"


def result_path(results_dir: Path, name: str) -> Path:
    return Path(results_dir) / f"{name}.json"


def ensure_new(path: Path) -> None:
    if path.exists():
        raise ResultExists(
            f"{path} already exists and results are never overwritten; move or rename it to run again"
        )


def select_cases(items: list[Item], num_cases: int | None, include_secondary: bool) -> list[Item]:
    """Primary items in loader order, the first num_cases of them (all when None). Secondary items
    (LoCoMo adversarial, MemoryAgentBench exploratory competencies) join only on request."""
    chosen = [i for i in items if i.primary or include_secondary]
    if num_cases is not None:
        if num_cases < 1:
            raise ValueError("num_cases must be at least 1")
        chosen = chosen[:num_cases]
    return chosen


def _case_view(n: int, item: Item, record: dict[str, Any] | None, error: dict[str, Any] | None) -> dict[str, Any]:
    base: dict[str, Any] = {
        "case": n,
        "item_id": item.item_id,
        "history_id": item.history_id,
        "category": item.category,
        "question": item.question,
        "gold": record["gold_answers"] if record else [item.gold],
    }
    if record is None:
        return {**base, "status": "error", "error": error or {"error": "no result recorded"}}
    reader, judge = record["reader"], record.get("judge")
    judge_usd = judge["cost_usd"] if judge else 0.0
    return {
        **base,
        "status": "ok",
        "answer": record["model_answer"],
        "correct": record.get("correct"),
        "metrics": record["metrics"],
        "context_tokens": record["context_tokens_counted"],
        "items_in_context": len(record["retrieved"]),
        "tokens": {
            "input": reader["usage"]["input_tokens"],
            "cached": reader["usage"]["cached_tokens"],
            "output": reader["usage"]["output_tokens"],
            "reasoning": reader["usage"]["reasoning_tokens"],
        },
        "cost_usd": {
            "reader": reader["cost_usd"],
            "judge": judge_usd,
            "retrieval": record["retrieval"]["cost"]["usd"],
            "total": reader["cost_usd"] + judge_usd + record["retrieval"]["cost"]["usd"],
        },
        "latency_ms": {
            "retrieval": record["latency_s"]["retrieval"] * 1000,
            "reader": record["latency_s"]["reader"] * 1000,
            "total": record["latency_s"]["query_total"] * 1000,
        },
        "judge_raw": judge["raw"] if judge else None,
    }


def build_report(
    *,
    name: str,
    system: str,
    arm: str,
    eval_name: str,
    num_cases: int | None,
    stage: str,
    budget_stage: str,
    cases: list[Item],
    records: list[dict[str, Any]],
    ingests: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    summary: RunSummary,
    cfg: dict[str, Any],
    cfg_hash: str,
    started_at: float,
    finished_at: float,
) -> dict[str, Any]:
    flat = summarize_records(records, ingests)
    by_item = {r["item_id"]: r for r in records}
    last_error = {e["item_id"]: e for e in errors if "item_id" in e}
    n = flat["n_items"]
    per_case = {k: (v / n if n else None) for k, v in flat["reader_tokens"].items()}
    return {
        "run": {
            "name": name,
            "memory_system": system,
            "arm": arm,
            "eval": eval_name,
            "num_cases": num_cases if num_cases is not None else "all",
            "stage": stage,
            "budget_stage": budget_stage,
            "started_at": datetime.fromtimestamp(started_at, UTC).isoformat(timespec="seconds"),
            "finished_at": datetime.fromtimestamp(finished_at, UTC).isoformat(timespec="seconds"),
            "wall_seconds": finished_at - started_at,
            "reader_model": cfg["reader"]["model"],
            "judge_model": cfg["judge"]["model"],
            "retrieval_token_budget": cfg["retrieval"]["token_budget"] if arm != "A" else None,
            "config_hash": cfg_hash,
            "versions": versions(),
        },
        "summary": {
            "cases": {
                "selected": len(cases),
                "completed": flat["n_items"],
                "errored": len(cases) - flat["n_items"],
                "does_not_fit": summary.does_not_fit,
                "judge_unparseable": flat["n_judge_unparseable"],
            },
            "accuracy": {
                "judge": flat["accuracy"],
                "exact_match": flat["answer_metrics"]["exact_match"],
                "substring_match": flat["answer_metrics"]["substring_match"],
                "token_f1": flat["answer_metrics"]["token_f1"],
                "gold_in_context": flat["answer_metrics"]["gold_in_context"],
                "scored_cases": flat["answer_metrics"]["n_scored"],
            },
            "by_category": flat["by_category"],
            "tokens": {
                "reader_total": flat["reader_tokens"],
                "reader_mean_per_case": per_case,
                "cache_hit_rate": flat["cache_hit_rate"],
                "context": flat["context_tokens"],
                "answer_mean": flat["answer_tokens_mean"],
                "judge_total": flat["judge_tokens"],
            },
            "retrieval": flat["retrieval"],
            "latency_ms": flat["latency_ms"],
            "cost_usd": flat["cost_usd"],
            "ingest": flat["ingest"],
            "models_returned": flat["models_returned"],
        },
        "cases": [
            _case_view(i, c, by_item.get(c.item_id), last_error.get(c.item_id))
            for i, c in enumerate(cases, start=1)
        ],
        "metric_notes": METRIC_NOTES,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as fh:  # raises FileExistsError instead of overwriting
        json.dump(report, fh, indent=2)
        fh.write("\n")


def execute_run(
    *,
    system: str,
    eval_name: str,
    num_cases: int | None,
    arm: Arm,
    cases: list[Item],
    histories: dict[str, History],
    reader: Reader,
    judge: Judge,
    stage: str,
    budget_stage: str,
    results_dir: Path,
    cfg: dict[str, Any],
    cfg_hash: str,
) -> Path:
    """Run arm over cases and write the result file. stage labels the run (smoke or chat);
    budget_stage is the spend cap it counts against. Raises ResultExists before any paid call
    when the file is already there. Raw records are kept if the run stops early; running the
    same command again resumes from them and then writes the file."""
    name = result_name(system, eval_name, num_cases)
    out = result_path(results_dir, name)
    ensure_new(out)
    store = ResultStore(Path(results_dir) / "runs" / name)
    run_id = f"{name}-{uuid.uuid4().hex[:8]}"
    started = time.time()
    summary = run_chat(
        arm=arm,
        items=cases,
        histories=histories,
        reader=reader,
        judge=judge,
        store=store,
        stage=budget_stage,
        run_id=run_id,
    )
    finalize_run(
        store,
        run_id=run_id,
        arm=arm.name,
        bench=eval_name,
        stage=stage,
        cfg_hash=cfg_hash,
        summary=summary,
    )
    selected = {c.item_id for c in cases}
    report = build_report(
        name=name,
        system=system,
        arm=arm.name,
        eval_name=eval_name,
        num_cases=num_cases,
        stage=stage,
        budget_stage=budget_stage,
        cases=cases,
        records=[r for r in store.read_items(arm.name, eval_name) if r["item_id"] in selected],
        ingests=store.read_ingest(arm.name, eval_name),
        errors=store.read_errors(arm.name, eval_name),
        summary=summary,
        cfg=cfg,
        cfg_hash=cfg_hash,
        started_at=started,
        finished_at=time.time(),
    )
    write_report(out, report)
    return out
