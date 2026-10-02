"""Resume-safe chat runner.

Items that already have a raw file are skipped, so a restarted run never repeats paid work and
never overwrites a result. A history that does not fit the window is recorded and skipped; it is
never truncated.
"""

from __future__ import annotations

import importlib.metadata
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import openai

from memstudy.arms.base import Arm, ArmContext
from memstudy.budget import BudgetExceeded
from memstudy.judge import Judge
from memstudy.llm import CallResult, IncompleteResponse
from memstudy.reader import Reader
from memstudy.schema import History, Item
from memstudy.store import ResultStore
from memstudy.tokens import DoesNotFit

HANDLED = (IncompleteResponse, openai.APIError)


@dataclass
class RunSummary:
    run_id: str
    completed: int = 0
    skipped_existing: int = 0
    errors: int = 0
    does_not_fit: list[str] = field(default_factory=list)


def build_record(
    *,
    run_id: str,
    arm: Arm,
    item: Item,
    ctx: ArmContext,
    read: CallResult,
    judged: tuple[bool | None, CallResult] | None,
) -> dict[str, Any]:
    u = read.usage
    record: dict[str, Any] = {
        "run_id": run_id,
        "arm": arm.name,
        "bench": item.bench,
        "item_id": item.item_id,
        "history_id": item.history_id,
        "category": item.category,
        "primary": item.primary,
        "abstention": bool(item.meta.get("abstention", False)),
        "question": item.question,
        "gold": item.gold,
        "model_answer": read.text,
        "context_tokens_counted": ctx.context_tokens,
        "retrieved": ctx.retrieved,
        "reader": {
            "model_returned": read.model_returned,
            "response_id": read.response_id,
            "usage": u.model_dump(),
            "cached_tokens": u.cached_tokens,
            "cache_hit_rate": (u.cached_tokens / u.input_tokens) if u.input_tokens else 0.0,
            "cost_usd": read.cost_usd,
            "warnings": read.warnings,
        },
        "retrieval": {
            "seconds": ctx.retrieval_seconds,
            "cost": ctx.retrieval_cost.to_dict(),
        },
        "latency_s": {
            "retrieval": ctx.retrieval_seconds,
            "reader": read.latency_s,
            "query_total": ctx.retrieval_seconds + read.latency_s,
        },
        "ts": time.time(),
    }
    if judged is not None:
        correct, call = judged
        record["judge"] = {
            "model_returned": call.model_returned,
            "raw": call.text,
            "usage": call.usage.model_dump(),
            "reasoning_tokens": call.usage.reasoning_tokens,
            "cost_usd": call.cost_usd,
            "latency_s": call.latency_s,
        }
        record["correct"] = correct
    return record


def run_chat(
    *,
    arm: Arm,
    items: list[Item],
    histories: dict[str, History],
    reader: Reader,
    judge: Judge | None,
    store: ResultStore,
    stage: str,
    run_id: str,
) -> RunSummary:
    summary = RunSummary(run_id=run_id)
    by_history: dict[str, list[Item]] = {}
    for item in items:
        by_history.setdefault(item.history_id, []).append(item)

    for history_id, group in by_history.items():
        bench = group[0].bench
        pending = [i for i in group if not store.has_item(arm.name, bench, i.item_id)]
        summary.skipped_existing += len(group) - len(pending)
        if not pending:
            continue
        history = histories[history_id]
        try:
            ingest = arm.prepare(history)
        except DoesNotFit as err:
            for item in pending:
                store.write_error(
                    arm.name,
                    bench,
                    item.item_id,
                    {"error": "does_not_fit", "tokens": err.tokens, "limit": err.limit},
                )
            summary.does_not_fit.append(history_id)
            summary.errors += len(pending)
            continue
        if ingest is not None and not store.has_ingest(arm.name, bench, history_id):
            store.write_ingest(
                arm.name,
                bench,
                history_id,
                {
                    "run_id": run_id,
                    "arm": arm.name,
                    "history_id": history_id,
                    "seconds": ingest.seconds,
                    "cost": ingest.cost.to_dict(),
                    "stored_units": ingest.stored_units,
                },
            )
        for item in pending:
            try:
                ctx = arm.context(item, history)
                read = reader.answer(
                    stage=stage,
                    tag={"arm": arm.name, "item": item.item_id},
                    context=ctx.text,
                    question=item.question,
                    question_date=item.question_date,
                    cache_prefix=ctx.cache_prefix,
                    cache_key=f"{arm.name}-{history.user_id}",
                )
                judged = None
                if judge is not None:
                    result = judge.grade(
                        stage=stage,
                        tag={"arm": arm.name, "item": item.item_id},
                        item=item,
                        model_answer=read.text,
                    )
                    judged = (result.correct, result.call)
                record = build_record(
                    run_id=run_id, arm=arm, item=item, ctx=ctx, read=read, judged=judged
                )
                store.write_item(arm.name, bench, item.item_id, record)
                summary.completed += 1
            except BudgetExceeded:
                raise
            except HANDLED as err:
                store.write_error(
                    arm.name,
                    bench,
                    item.item_id,
                    {"error": type(err).__name__, "message": str(err)},
                )
                summary.errors += 1
    return summary


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def versions() -> dict[str, Any]:
    pkgs: dict[str, str | None] = {}
    for name in ("openai", "mem0ai", "tiktoken", "memstudy", "numpy"):
        try:
            pkgs[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pkgs[name] = None
    return {"python": platform.python_version(), "packages": pkgs, "git_commit": _git_commit()}


def summarize_records(
    records: list[dict[str, Any]], ingests: list[dict[str, Any]]
) -> dict[str, Any]:
    """Token, latency, and cost totals for a set of raw records (always derived from files)."""
    tot = {k: 0 for k in ("input", "cached", "cache_write", "output", "reasoning")}
    judge_tot = {"input": 0, "output": 0, "reasoning": 0}
    reader_usd = judge_usd = retrieval_usd = 0.0
    latencies: list[float] = []
    models: set[str] = set()
    for r in records:
        u = r["reader"]["usage"]
        tot["input"] += u["input_tokens"]
        tot["cached"] += u["cached_tokens"]
        tot["cache_write"] += u["cache_write_tokens"]
        tot["output"] += u["output_tokens"]
        tot["reasoning"] += u["reasoning_tokens"]
        reader_usd += r["reader"]["cost_usd"]
        retrieval_usd += r["retrieval"]["cost"]["usd"]
        latencies.append(r["latency_s"]["query_total"])
        models.add(r["reader"]["model_returned"])
        if "judge" in r:
            ju = r["judge"]["usage"]
            judge_tot["input"] += ju["input_tokens"]
            judge_tot["output"] += ju["output_tokens"]
            judge_tot["reasoning"] += ju["reasoning_tokens"]
            judge_usd += r["judge"]["cost_usd"]
            models.add(r["judge"]["model_returned"])
    n = len(records)
    graded = [r for r in records if r.get("correct") is not None]
    ordered = sorted(latencies)
    return {
        "n_items": n,
        "n_graded": len(graded),
        "accuracy": (sum(1 for r in graded if r["correct"]) / len(graded)) if graded else None,
        "reader_tokens": tot,
        "cache_hit_rate": (tot["cached"] / tot["input"]) if tot["input"] else 0.0,
        "judge_tokens": judge_tot,
        "latency_s": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": ordered[n // 2] if n else None,
            "p95": ordered[min(n - 1, int(0.95 * n))] if n else None,
        },
        "cost_usd": {
            "reader": reader_usd,
            "judge": judge_usd,
            "retrieval": retrieval_usd,
            "ingest": sum(i["cost"]["usd"] for i in ingests),
        },
        "ingest": {
            "histories": len(ingests),
            "seconds": sum(i["seconds"] for i in ingests),
            "calls": sum(i["cost"]["calls"] for i in ingests),
        },
        "models_returned": sorted(models),
    }


def finalize_run(
    store: ResultStore,
    *,
    run_id: str,
    arm: str,
    bench: str,
    stage: str,
    cfg_hash: str,
    summary: RunSummary,
) -> dict[str, Any]:
    records = [r for r in store.read_items(arm, bench) if r["run_id"] == run_id]
    ingests = [i for i in store.read_ingest(arm, bench) if i["run_id"] == run_id]
    run = {
        "run_id": run_id,
        "arm": arm,
        "bench": bench,
        "stage": stage,
        "config_hash": cfg_hash,
        "versions": versions(),
        "completed": summary.completed,
        "skipped_existing": summary.skipped_existing,
        "errors": summary.errors,
        "does_not_fit": summary.does_not_fit,
        **summarize_records(records, ingests),
    }
    store.write_run(run_id, run)
    return run
