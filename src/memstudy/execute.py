"""One named run: pick cases, run an arm over them, write one readable result file.

    python -m memstudy run --memory_system mem0 --eval locomo --num_cases 1

writes results/mem0_locomo_<YYYYMMDD_HHMMSS>.json: the run's provenance, a summary, and every
question with its gold answer, the model's answer, the verdict and the scores. It is the only
file a run creates (besides the shared spend ledger); the file is created exclusively and never
overwritten. If a run stops early the file is still written, marked incomplete, with every
question answered so far.

A case is one history: a LoCoMo conversation (with all of its questions), a LongMemEval
question's haystack, a MemoryAgentBench document.
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from memstudy.arms.base import Arm, IngestStats
from memstudy.judge import Judge
from memstudy.prompts import TASK_PROMPTS
from memstudy.reader import Reader
from memstudy.runner import RunSummary, _dist, run_chat, summarize_records, versions
from memstudy.schema import History, Item
from memstudy.store import MemoryStore

SYSTEMS = {"no_memory": "A", "mem0": "B", "rag": "C", "supermemory": "D"}
EVALS = ("locomo", "longmemeval", "memoryagentbench", "composite")

METRIC_NOTES = {
    "accuracy.judge": "Share of cases the judge (GPT-5 nano) graded CORRECT (the pre-registered primary metric).",
    "accuracy.exact_match": "Normalized answer equals a normalized accepted answer.",
    "accuracy.substring_match": "A normalized accepted answer appears inside the normalized answer.",
    "accuracy.token_f1": "Mean best token-overlap F1 against the accepted answers.",
    "grading": "The grader depends on the item, not the arm. LongMemEval questions use LongMemEval's "
    "own per-type prompts (temporal: off-by-one allowed; knowledge-update: old plus new value is "
    "fine; preference: the gold is a rubric; unanswerable: must say so). InfBench summaries are "
    "scored, not judged (secondary.summary_*). Everything else uses the study's single judge rule.",
    "secondary.accuracy_mem0_judge": "LoCoMo categories 1-4 only: Mem0's own judge prompt (lenient: "
    "correct when it touches the gold's topic), graded by the same nano judge. For comparison with "
    "published Mem0 and Zep numbers only; never the primary metric.",
    "secondary.native_match": "MemoryAgentBench only: the benchmark's own pass test per source "
    "(icl: parsed label equals the gold; detective_qa: same option letter; others: accepted answer "
    "appears in the answer).",
    "secondary.summary_f1": "InfBench Sum only: MemoryAgentBench's summary grade, fluency x harmonic "
    "mean of keypoint recall and sentence precision.",
    "accuracy.gold_in_context": "Share of cases where an accepted answer appears verbatim in the "
    "context the reader saw (retrieval proxy; short answers can match by chance).",
    "stage_attribution": "Memory arms only. not_stored: no accepted answer appears verbatim in the "
    "store (extraction, a lower bound because facts paraphrase); stored_not_retrieved: stored but "
    "absent from the reader's context (retrieval); stored_retrieved_correct / "
    "stored_retrieved_graded: judge accuracy once the gold was in context (reading).",
    "scored_cases": "Abstention cases have no answer string, so only the judge scores them and "
    "they are left out of exact_match, substring_match, token_f1 and gold_in_context.",
    "cost_usd": "Reader, judge, retrieval and ingestion spend, from the provider's reported "
    "usage and the verified prices in configs/prices.yaml.",
    "time": "Seconds. per_question_s is retrieval + reader + judge for one question; "
    "wall_per_question_s is total wall time (ingestion included) divided by questions.",
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


def result_name(system: str, eval_name: str, when: datetime | None = None) -> str:
    """system_eval_YYYYMMDD_HHMMSS (local time)."""
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{system}_{eval_name}_{stamp}"


def result_path(results_dir: Path, name: str) -> Path:
    return Path(results_dir) / f"{name}.json"


def ensure_new(path: Path) -> None:
    if path.exists():
        raise ResultExists(
            f"{path} already exists and results are never overwritten; move or rename it to run again"
        )


# MemoryAgentBench sources never run, for any arm (PREREG amendment A3).
EXCLUDED_SOURCES = ("infbench_sum",)


def _excluded(item: Item) -> bool:
    return str(item.meta.get("source", "")).startswith(EXCLUDED_SOURCES)


def select_cases(items: list[Item], num_cases: int | None, include_secondary: bool) -> list[Item]:
    """Primary items in loader order, limited to the first num_cases histories (all when None),
    with every selected question of each of those histories. Secondary items (LoCoMo adversarial,
    MemoryAgentBench exploratory competencies) join only on request. Excluded sources never join."""
    chosen = [i for i in items if (i.primary or include_secondary) and not _excluded(i)]
    if num_cases is not None:
        if num_cases < 1:
            raise ValueError("num_cases must be at least 1")
        keep = list(dict.fromkeys(i.history_id for i in chosen))[:num_cases]
        chosen = [i for i in chosen if i.history_id in keep]
    return chosen


def retry_cases(items: list[Item], result_file: Path, system: str, eval_name: str) -> list[Item]:
    """Items an earlier result file has no answer for (errored or not run), in loader order.
    Histories that did not fit the window are left out: a retry cannot change that."""
    result = json.loads(result_file.read_text())
    run = result["run"]
    if (run["memory_system"], run["eval"]) != (system, eval_name):
        raise SystemExit(
            f"{result_file} is a {run['memory_system']} run on {run['eval']}, "
            f"not {system} on {eval_name}"
        )
    failed = {
        c["item_id"]
        for c in result["cases"]
        if c["status"] != "ok" and c.get("error", {}).get("error") != "does_not_fit"
    }
    return [i for i in items if i.item_id in failed and not _excluded(i)]


def filter_sources(items: list[Item], prefixes: list[str]) -> list[Item]:
    """MemoryAgentBench items whose source sub-dataset starts with one of the prefixes."""
    return [i for i in items if str(i.meta.get("source", "")).startswith(tuple(prefixes))]


def apply_task_prompts(items: list[Item], letter: str) -> list[Item]:
    """Wrap each question whose source has a benchmark task template (TASK_PROMPTS) in that
    template; other items are unchanged. Arm A gets the long-context template, the rest RAG."""
    agent = "long_context_agent" if letter == "A" else "rag_agent"
    out = []
    for item in items:
        source = str(item.meta.get("source", ""))
        prefix = next((p for p in TASK_PROMPTS if source.startswith(p)), None)
        if prefix is None:
            out.append(item)
            continue
        question = TASK_PROMPTS[prefix][agent].format(question=item.question)
        meta = {**item.meta, "task_prompt": prefix, "raw_question": item.question}
        out.append(item.model_copy(update={"question": question, "meta": meta}))
    return out


def _verdict(correct: bool | None, judged: bool) -> str:
    if not judged:
        return "n/a"
    return {True: "CORRECT", False: "INCORRECT", None: "UNPARSEABLE"}[correct]


def _judge_text(record: dict[str, Any], judged: bool) -> str:
    """The verdict, or for a summary (scored, not judged) its summary f1."""
    if judged and record["judge"].get("scheme") == "summary":
        scores = record.get("summary_scores")
        return f"summary f1 {scores['f1']:.3f}" if scores else "summary UNSCORED"
    return _verdict(record.get("correct"), judged)


def _f1_text(metrics: dict[str, Any] | None) -> str:
    return f"{metrics['token_f1']:.3f}" if metrics else "n/a"


def _wrapped(label: str, text: str, width: int = 100) -> str:
    return textwrap.fill(
        " ".join(text.split()) or "(empty)",
        width=width,
        initial_indent=f"{label:<15}",
        subsequent_indent=" " * 15,
    )


def _seconds(record: dict[str, Any]) -> float:
    judge = record["judge"]["latency_s"] if "judge" in record else 0.0
    return float(record["latency_s"]["query_total"] + judge)


class LivePrinter:
    """Prints each question as the run works through it, with the gold answer, the given answer,
    the judge verdict and the token F1, and a running tally."""

    def __init__(self, total: int, rule: str = "_" * 100) -> None:
        self.total, self.rule = total, rule
        self.done = self.correct = self.graded = 0

    @staticmethod
    def _out(text: str) -> None:
        print(text, flush=True)

    def history_started(self, history_id: str, questions: int) -> None:
        self._out(f"\n=== {history_id}: {questions} questions ===")

    def ingested(self, history_id: str, ingest: IngestStats) -> None:
        self._out(
            f"ingested {history_id}: {ingest.stored_units} stored units, {ingest.seconds:.1f}s, "
            f"${ingest.cost.usd:.4f}"
        )

    def answered(self, item: Item, record: dict[str, Any]) -> None:
        self.done += 1
        if record.get("correct") is not None:
            self.graded += 1
            self.correct += bool(record["correct"])
        judged = "judge" in record
        running = f"{self.correct}/{self.graded} correct" if self.graded else "-"
        self._out(
            "\n".join(
                [
                    f"\n[{self.done}/{self.total}] {item.item_id} ({item.category})",
                    _wrapped("question:", item.question),
                    _wrapped("actual answer:", " | ".join(record["gold_answers"])),
                    _wrapped("given answer:", record["model_answer"]),
                    f"judge: {_judge_text(record, judged)}   "
                    f"f1: {_f1_text(record['metrics'])}   time: {_seconds(record):.1f}s   "
                    f"running: {running}",
                    self.rule,
                ]
            )
        )

    def failed(self, item: Item, error: dict[str, Any]) -> None:
        self.done += 1
        self._out(f"\n[{self.done}/{self.total}] {item.item_id} ({item.category}) FAILED")
        self._out(_wrapped("question:", item.question))
        self._out(_wrapped("error:", f"{error.get('error')}: {error.get('message', '')}".strip(": ")))
        self._out(self.rule)


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
        if error is None:
            return {**base, "status": "not_run"}
        return {**base, "status": "error", "error": error}
    reader, judge = record["reader"], record.get("judge")
    metrics = record["metrics"] or {}
    judge_usd = (judge["cost_usd"] if judge else 0.0) + record.get("judge_mem0", {}).get("cost_usd", 0.0)
    return {
        **base,
        "status": "ok",
        "answer": record["model_answer"],
        "correct": record.get("correct"),
        "correct_mem0": record.get("correct_mem0"),
        "native_match": metrics.get("native_match"),
        "summary_scores": record.get("summary_scores"),
        "token_f1": metrics.get("token_f1"),
        "exact_match": metrics.get("exact_match"),
        "substring_match": metrics.get("substring_match"),
        "gold_in_context": metrics.get("gold_in_context"),
        "gold_in_store": record.get("gold_in_store"),
        "seconds": {
            "retrieval": record["latency_s"]["retrieval"],
            "reader": record["latency_s"]["reader"],
            "judge": judge["latency_s"] if judge else 0.0,
            "total": _seconds(record),
        },
        "tokens": {
            "context": record["context_tokens_counted"],
            "input": reader["usage"]["input_tokens"],
            "cached": reader["usage"]["cached_tokens"],
            "output": reader["usage"]["output_tokens"],
            "reasoning": reader["usage"]["reasoning_tokens"],
        },
        "items_in_context": len(record["retrieved"]),
        "cost_usd": reader["cost_usd"] + judge_usd + record["retrieval"]["cost"]["usd"],
        "judge_raw": judge["raw"] if judge else None,
    }


def _tidy(value: Any, key: str = "") -> Any:
    """Round floats so the file reads cleanly: 4 places, 6 for dollar amounts."""
    if isinstance(value, float):
        return round(value, 6 if "usd" in key else 4)
    if isinstance(value, dict):
        return {k: _tidy(v, k if "usd" in k else key) for k, v in value.items()}
    if isinstance(value, list):
        return [_tidy(v, key) for v in value]
    return value


def _time_summary(
    records: list[dict[str, Any]], ingests: list[dict[str, Any]], wall: float
) -> dict[str, Any]:
    n = len(records)
    judged = [r["judge"]["latency_s"] for r in records if "judge" in r]
    return {
        "total_wall_s": wall,
        "ingest_s": sum(i["seconds"] for i in ingests),
        "wall_per_question_s": (wall / n) if n else None,
        "per_question_s": _dist([_seconds(r) for r in records]),
        "retrieval_s": _dist([r["latency_s"]["retrieval"] for r in records]),
        "reader_s": _dist([r["latency_s"]["reader"] for r in records]),
        "judge_s": _dist(judged),
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
    status: str = "complete",
    workers: int = 1,
) -> dict[str, Any]:
    flat = summarize_records(records, ingests)
    by_item = {r["item_id"]: r for r in records}
    last_error = {e["item_id"]: e for e in errors if "item_id" in e}
    views = [
        _case_view(i, c, by_item.get(c.item_id), last_error.get(c.item_id))
        for i, c in enumerate(cases, start=1)
    ]
    n = flat["n_items"]
    per_case = {k: (v / n if n else None) for k, v in flat["reader_tokens"].items()}
    graded = [v for v in views if v.get("correct") is not None]
    report: dict[str, Any] = {
        "run": {
            "name": name,
            "status": status,
            "memory_system": system,
            "arm": arm,
            "eval": eval_name,
            "num_cases": num_cases if num_cases is not None else "all",
            "stage": stage,
            "budget_stage": budget_stage,
            "parallel_questions": workers,
            "started_at": datetime.fromtimestamp(started_at).astimezone().isoformat(timespec="seconds"),
            "finished_at": datetime.fromtimestamp(finished_at).astimezone().isoformat(timespec="seconds"),
            "reader_model": cfg["reader"]["model"],
            "judge_model": cfg["judge"]["model"],
            "retrieval_token_budget": cfg["retrieval"]["token_budget"] if arm != "A" else None,
            "config_hash": cfg_hash,
            "versions": versions(),
        },
        "summary": {
            "cases": {
                "conversations": len({c.history_id for c in cases}),
                "questions": len(cases),
                "answered": n,
                "errored": sum(1 for v in views if v["status"] == "error"),
                "not_run": sum(1 for v in views if v["status"] == "not_run"),
                "does_not_fit": summary.does_not_fit,
                "judge_unparseable": flat["n_judge_unparseable"],
                "mem0_judge_unparseable": flat["n_mem0_judge_unparseable"],
                "summary_unparseable": flat["n_summary_unparseable"],
                "incorrect_cases": [v["case"] for v in views if v.get("correct") is False],
                "errored_cases": [v["case"] for v in views if v["status"] == "error"],
            },
            "accuracy": {
                "judge": flat["accuracy"],
                "judge_correct": sum(1 for v in graded if v["correct"]),
                "judge_graded": len(graded),
                "token_f1": flat["answer_metrics"]["token_f1"],
                "exact_match": flat["answer_metrics"]["exact_match"],
                "substring_match": flat["answer_metrics"]["substring_match"],
                "gold_in_context": flat["answer_metrics"]["gold_in_context"],
                "scored_cases": flat["answer_metrics"]["n_scored"],
            },
            "secondary": flat["secondary"],
            "by_category": flat["by_category"],
            "time": _time_summary(records, ingests, finished_at - started_at),
            "tokens": {
                "reader_total": flat["reader_tokens"],
                "reader_mean_per_case": per_case,
                "cache_hit_rate": flat["cache_hit_rate"],
                "context": flat["context_tokens"],
                "answer_mean": flat["answer_tokens_mean"],
                "judge_total": flat["judge_tokens"],
            },
            "cost_usd": flat["cost_usd"],
            "retrieval": flat["retrieval"],
            "ingest": flat["ingest"],
            "stage_attribution": flat["stage_attribution"],
            "models_returned": flat["models_returned"],
        },
        "cases": views,
        "metric_notes": METRIC_NOTES,
    }
    return _tidy(report)  # type: ignore[no-any-return]


def format_summary(report: dict[str, Any]) -> str:
    """The end-of-run summary as plain text for the terminal."""
    run, s = report["run"], report["summary"]
    c, a, t, tok, cost = s["cases"], s["accuracy"], s["time"], s["tokens"], s["cost_usd"]

    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{100 * v:.1f}%"

    def num(v: float | None, fmt: str = ".3f") -> str:
        return "n/a" if v is None else format(v, fmt)

    per_q = t["per_question_s"]
    lines = [
        "",
        "=" * 100,
        f"SUMMARY  {run['memory_system']} on {run['eval']}  ({run['status']})",
        "=" * 100,
        f"conversations {c['conversations']} | questions {c['questions']} | answered {c['answered']}"
        f" | errored {c['errored']} | not run {c['not_run']}",
        "",
        f"judge accuracy   {pct(a['judge'])}  ({a['judge_correct']}/{a['judge_graded']})",
        f"token F1         {num(a['token_f1'])}",
        f"exact match      {pct(a['exact_match'])}",
        f"substring match  {pct(a['substring_match'])}",
        f"gold in context  {pct(a['gold_in_context'])}",
    ]
    sec = s["secondary"]
    if sec["n_mem0_graded"]:
        lines.append(f"mem0 judge       {pct(sec['accuracy_mem0_judge'])}  (secondary, {sec['n_mem0_graded']} graded)")
    if sec["n_native"]:
        lines.append(f"native match     {pct(sec['native_match'])}  ({sec['n_native']} scored)")
    if sec["n_summaries_scored"]:
        lines.append(f"summary f1       {num(sec['summary_f1'])}  ({sec['n_summaries_scored']} scored)")
    lines += [
        "",
        "by category      n     judge     f1",
    ]
    for cat, g in s["by_category"].items():
        lines.append(f"  {cat:<12}{g['n']:>4}  {pct(g['accuracy_judge']):>8}  {num(g['token_f1']):>6}")
    lines += [
        "",
        f"time   total {t['total_wall_s']:.1f}s | ingest {t['ingest_s']:.1f}s | "
        f"per question mean {num(per_q['mean'], '.2f')}s p50 {num(per_q['p50'], '.2f')}s "
        f"p95 {num(per_q['p95'], '.2f')}s | wall/question {num(t['wall_per_question_s'], '.2f')}s",
        f"tokens per question   input {num(tok['reader_mean_per_case']['input'], '.0f')} | "
        f"output {num(tok['answer_mean'], '.1f')} | context mean {num(tok['context']['mean'], '.0f')} "
        f"| cache hit {pct(tok['cache_hit_rate'])}",
        f"cost   total ${cost['total']:.4f} | per question ${num(cost['per_item'], '.5f')} | "
        f"reader ${cost['reader']:.4f} judge ${cost['judge']:.4f} "
        f"retrieval ${cost['retrieval']:.4f} ingest ${cost['ingest']:.4f}",
    ]
    if c["incorrect_cases"]:
        lines.append(f"incorrect cases   {', '.join(str(n) for n in c['incorrect_cases'])}")
    if c["errored_cases"]:
        lines.append(f"errored cases     {', '.join(str(n) for n in c['errored_cases'])}")
    return "\n".join(lines)


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
    name: str | None = None,
    progress: LivePrinter | None = None,
    workers: int = 1,
) -> Path:
    """Run arm over cases and write the one result file. stage labels the run (smoke or chat);
    budget_stage is the spend cap it counts against. Raises ResultExists before any paid call
    when the file is already there. If the run stops early (budget cap, API error, Ctrl-C) the
    file is still written, marked incomplete, and the error is re-raised."""
    name = name or result_name(system, eval_name)
    out = result_path(results_dir, name)
    ensure_new(out)
    store = MemoryStore()
    run_id = f"{name}-{uuid.uuid4().hex[:8]}"
    started = time.time()
    summary = RunSummary(run_id=run_id)
    status = "complete"
    failure: BaseException | None = None
    try:
        summary = run_chat(
            arm=arm,
            items=cases,
            histories=histories,
            reader=reader,
            judge=judge,
            store=store,
            stage=budget_stage,
            run_id=run_id,
            progress=progress,
            workers=workers,
        )
    except BaseException as err:  # keep the answers already paid for
        status = f"incomplete: {type(err).__name__}: {err}"
        failure = err
    report = build_report(
        name=name,
        system=system,
        arm=arm.name,
        eval_name=eval_name,
        num_cases=num_cases,
        stage=stage,
        budget_stage=budget_stage,
        cases=cases,
        records=store.read_items(arm.name, eval_name),
        ingests=store.read_ingest(arm.name, eval_name),
        errors=store.read_errors(arm.name, eval_name),
        summary=summary,
        cfg=cfg,
        cfg_hash=cfg_hash,
        started_at=started,
        finished_at=time.time(),
        status=status,
        workers=workers,
    )
    write_report(out, report)
    if failure is not None:
        print(f"\nrun stopped early; answers so far are in {out}", file=sys.stderr, flush=True)
        raise failure
    return out
