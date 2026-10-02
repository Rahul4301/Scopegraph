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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Protocol

import openai

from memstudy.arms.base import Arm, ArmContext, IngestStats
from memstudy.budget import BudgetExceeded
from memstudy.judge import Judge
from memstudy.llm import CallResult, IncompleteResponse
from memstudy.prompts import accepted_answers
from memstudy.reader import Reader
from memstudy.schema import History, Item
from memstudy.scoring import answer_metrics, gold_in_store
from memstudy.store import ResultStore
from memstudy.tokens import DoesNotFit

HANDLED = (IncompleteResponse, openai.APIError)


class Progress(Protocol):
    """Optional live feedback while a run is going. Never changes what the run does."""

    def history_started(self, history_id: str, questions: int) -> None: ...

    def ingested(self, history_id: str, ingest: IngestStats) -> None: ...

    def answered(self, item: Item, record: dict[str, Any]) -> None: ...

    def failed(self, item: Item, error: dict[str, Any]) -> None: ...


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
        "gold_answers": accepted_answers(item),
        "model_answer": read.text,
        # Abstention items have no answer string to match, so only the judge scores them.
        "metrics": None
        if item.meta.get("abstention")
        else answer_metrics(read.text, ctx.text, accepted_answers(item)),
        # Stage attribution (arms B and D): was the gold fact stored? None when the arm has no
        # store, or for abstention items, which have no answer string to look for.
        "gold_in_store": None
        if ctx.stored_text is None or item.meta.get("abstention")
        else gold_in_store(ctx.stored_text, accepted_answers(item)),
        "context_tokens_counted": ctx.context_tokens,
        "retrieved": ctx.retrieved,
        "candidates": ctx.candidates,
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
    progress: Progress | None = None,
    workers: int = 1,
) -> RunSummary:
    summary = RunSummary(run_id=run_id)
    by_history: dict[str, list[Item]] = {}
    for item in items:
        by_history.setdefault(item.history_id, []).append(item)

    def run_history(history_id: str, group: list[Item]) -> None:
        bench = group[0].bench
        pending = [i for i in group if not store.has_item(arm.name, bench, i.item_id)]
        summary.skipped_existing += len(group) - len(pending)
        if not pending:
            return
        history = histories[history_id]
        if progress:
            progress.history_started(history_id, len(pending))
        try:
            ingest = arm.prepare(history)
        except DoesNotFit as err:
            for item in pending:
                error = {
                    "item_id": item.item_id,
                    "history_id": history_id,
                    "error": "does_not_fit",
                    "tokens": err.tokens,
                    "limit": err.limit,
                }
                store.write_error(arm.name, bench, item.item_id, error)
                if progress:
                    progress.failed(item, error)
            summary.does_not_fit.append(history_id)
            summary.errors += len(pending)
            return
        if ingest is not None and progress:
            progress.ingested(history_id, ingest)
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
        # One lock covers arm.context (the arms are not built for concurrent use), the counters
        # and the progress output; the slow reader and judge calls run outside it.
        lock = threading.Lock()

        def answer(item: Item) -> None:
            try:
                with lock:
                    ctx = arm.context(item, history)
                read = reader.answer(
                    stage=stage,
                    tag={"arm": arm.name, "item": item.item_id},
                    context=ctx.text,
                    question=item.question,
                    question_date=item.question_date,
                    cache_prefix=ctx.cache_prefix,
                    cache_key=f"{arm.name}-{history.store_key}",
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
                with lock:
                    summary.completed += 1
                    if progress:
                        progress.answered(item, record)
            except BudgetExceeded:
                raise
            except HANDLED as err:
                error = {
                    "item_id": item.item_id,
                    "history_id": history_id,
                    "error": type(err).__name__,
                    "message": str(err),
                }
                store.write_error(arm.name, bench, item.item_id, error)
                with lock:
                    summary.errors += 1
                    if progress:
                        progress.failed(item, error)

        if workers <= 1 or len(pending) == 1:
            for item in pending:
                answer(item)
            return
        answer(pending[0])  # first call alone, so the prompt cache is warm for the rest
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(answer, item) for item in pending[1:]]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    for history_id, group in by_history.items():
        run_history(history_id, group)
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


def _dist(values: list[float], scale: float = 1.0) -> dict[str, float | None]:
    """mean, p50, p95 and max of values (times scale); None throughout when there are none."""
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(v * scale for v in values)
    n = len(ordered)
    return {
        "mean": statistics.fmean(ordered),
        "p50": ordered[n // 2],
        "p95": ordered[min(n - 1, int(0.95 * n))],
        "max": ordered[-1],
    }


def _rate(flags: list[bool]) -> float | None:
    return (sum(flags) / len(flags)) if flags else None


def _group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [r for r in records if r.get("correct") is not None]
    scored = [r["metrics"] for r in records if r.get("metrics")]
    return {
        "n": len(records),
        "accuracy_judge": _rate([bool(r["correct"]) for r in graded]),
        "exact_match": _rate([m["exact_match"] for m in scored]),
        "substring_match": _rate([m["substring_match"] for m in scored]),
        "token_f1": statistics.fmean([m["token_f1"] for m in scored]) if scored else None,
    }


def stage_attribution(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Where memory-arm failures arise: extraction (gold not stored), retrieval (stored but not
    in the reader's context) or reading (in context but graded wrong). None for arms with no
    store. Gold-in-store and gold-in-context are verbatim tests, so both are lower bounds for
    systems that paraphrase; reading is judged by the sole grader."""
    rows = [r for r in records if r.get("gold_in_store") is not None and r.get("metrics")]
    if not rows:
        return None
    stored = [r for r in rows if r["gold_in_store"]]
    retrieved = [r for r in stored if r["metrics"]["gold_in_context"]]
    graded = [r for r in retrieved if r.get("correct") is not None]
    return {
        "n": len(rows),
        "stored": len(stored),
        "stored_and_retrieved": len(retrieved),
        "stored_retrieved_graded": len(graded),
        "stored_retrieved_correct": sum(1 for r in graded if r["correct"]),
        "not_stored": len(rows) - len(stored),
        "stored_not_retrieved": len(stored) - len(retrieved),
        "correct_overall": sum(1 for r in rows if r.get("correct")),
    }


def summarize_records(
    records: list[dict[str, Any]], ingests: list[dict[str, Any]]
) -> dict[str, Any]:
    """Accuracy, token, latency, and cost totals for a set of raw records (always derived from
    files). Latencies are reported in milliseconds, costs in USD."""
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
            if not r["judge"]["model_returned"].startswith("rule:"):
                models.add(r["judge"]["model_returned"])
    n = len(records)
    graded = [r for r in records if r.get("correct") is not None]
    ordered = sorted(latencies)
    ingest_usd = sum(i["cost"]["usd"] for i in ingests)
    total_usd = reader_usd + judge_usd + retrieval_usd + ingest_usd
    by_category: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_category.setdefault(r["category"], []).append(r)
    scored = [r["metrics"] for r in records if r.get("metrics")]
    return {
        "stage_attribution": stage_attribution(records),
        "n_items": n,
        "n_graded": len(graded),
        "n_judge_unparseable": sum(1 for r in records if "judge" in r and r.get("correct") is None),
        "accuracy": (sum(1 for r in graded if r["correct"]) / len(graded)) if graded else None,
        "answer_metrics": {
            "exact_match": _rate([m["exact_match"] for m in scored]),
            "substring_match": _rate([m["substring_match"] for m in scored]),
            "token_f1": statistics.fmean([m["token_f1"] for m in scored]) if scored else None,
            "gold_in_context": _rate([m["gold_in_context"] for m in scored]),
            "n_scored": len(scored),
        },
        "by_category": {c: _group_summary(rs) for c, rs in sorted(by_category.items())},
        "context_tokens": _dist([r["context_tokens_counted"] for r in records]),
        "answer_tokens_mean": (tot["output"] / n) if n else None,
        "retrieval": {
            "candidates_mean": statistics.fmean([r.get("candidates", 0) for r in records])
            if records
            else None,
            "items_in_context_mean": statistics.fmean([len(r["retrieved"]) for r in records])
            if records
            else None,
        },
        "reader_tokens": tot,
        "cache_hit_rate": (tot["cached"] / tot["input"]) if tot["input"] else 0.0,
        "judge_tokens": judge_tot,
        "latency_s": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": ordered[n // 2] if n else None,
            "p95": ordered[min(n - 1, int(0.95 * n))] if n else None,
        },
        "latency_ms": {
            "retrieval": _dist([r["latency_s"]["retrieval"] for r in records], 1000.0),
            "reader": _dist([r["latency_s"]["reader"] for r in records], 1000.0),
            "query_total": _dist(latencies, 1000.0),
        },
        "cost_usd": {
            "reader": reader_usd,
            "judge": judge_usd,
            "retrieval": retrieval_usd,
            "ingest": ingest_usd,
            "total": total_usd,
            "per_item": (total_usd / n) if n else None,
        },
        "ingest": {
            "histories": len(ingests),
            "seconds": sum(i["seconds"] for i in ingests),
            "calls": sum(i["cost"]["calls"] for i in ingests),
            "tokens": sum(i["cost"]["usage"]["input_tokens"] for i in ingests),
            "stored_units": sum(i["stored_units"] for i in ingests),
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
