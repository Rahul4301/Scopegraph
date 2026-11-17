"""RQ4: after how many questions per history are RAG (C) and Mem0 (B) cheaper than full context (A)?

Free and offline: reads finished result files, never calls an API. For each history it measures
the inputs of the closed-form cost model (costmodel.py) and solves for the break-even question
count, once with the full-context cache warm between questions and once with no cache:

- history tokens: arm A's context tokens for that history;
- question and answer tokens: means over arm A's questions on that history (or group);
- memory read: the memory arm's mean retrieved context tokens on the same questions, read
  uncached, plus embedding the question;
- ingestion: the memory arm's measured per-history ingestion cost. RAG's is recomputed exactly
  from its chunks when the result file predates per-history ingest records (embedding is a pure
  function of the chunk tokens).

Rows group histories into 100K-token buckets (`by="bucket"`) or questions by category
(`by="category"`). Each cell is the median break-even over histories with a percentile bootstrap
interval over histories (seed 0). A history with no crossover counts as infinitely many questions,
so a median of "none" means at least half the histories never break even.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from memstudy.budget import ModelPrice
from memstudy.costmodel import QueryShape, crossover_queries
from memstudy.execute import ingest_by_history

BUCKET = 100_000
MEMORY_ARMS = {"C": "RAG", "B": "Mem0"}
MODES = ("warm", "nocache")


def _ok(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in cases if c.get("status") == "ok" and c.get("tokens")]


def bucket_label(tokens: int) -> str:
    lo = (tokens // BUCKET) * BUCKET
    return f"{lo // 1000}K-{(lo + BUCKET) // 1000}K"


def _bucket_key(label: str) -> int:
    return int(label.split("K", 1)[0])


def break_even(
    reader: ModelPrice,
    embed: ModelPrice,
    history_tokens: int,
    ingest_usd: float,
    fc_cases: Sequence[dict[str, Any]],
    mem_cases: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Break-even question count for one history (or one group of its questions), per mode.
    math.inf when the memory arm is never cheaper per question."""
    question = statistics.fmean(c["tokens"]["input"] - c["tokens"]["context"] for c in fc_cases)
    answer = statistics.fmean(c["tokens"]["output"] for c in fc_cases)
    shape = QueryShape(question_tokens=round(question), answer_tokens=round(answer))
    context = statistics.fmean(c["tokens"]["context"] for c in mem_cases)
    retrieval = question * embed.input / 1e6
    out: dict[str, float] = {}
    for mode in MODES:
        q = crossover_queries(
            reader, history_tokens, ingest_usd, round(context), shape,
            cached=(mode == "warm"), retrieval_usd_per_query=retrieval,
        )
        out[mode] = math.inf if q is None else max(1.0, math.ceil(q))
    return out


def rag_ingest_usd(history: Any, chunk_tokens: int, embed: ModelPrice) -> float:
    """Arm C ingestion recomputed from its chunks: every chunk is embedded once."""
    from memstudy.arms.rag import chunk_history
    from memstudy.tokens import count_tokens

    return sum(count_tokens(c) for c in chunk_history(history, chunk_tokens)) * embed.input / 1e6


def rag_ingest_from(histories: dict[str, Any], chunk_tokens: int, embed: ModelPrice) -> Callable[[str], float]:
    """Arm C ingestion cost by history id, recomputed from the loaded histories."""
    return lambda history_id: rag_ingest_usd(histories[history_id], chunk_tokens, embed)


def _median_ci(values: Sequence[float], reps: int, seed: int) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(values)
    meds = sorted(statistics.median(rng.choices(values, k=n)) for _ in range(reps))
    return statistics.median(values), meds[int(0.025 * reps)], meds[min(reps - 1, int(0.975 * reps))]


def per_history_rows(
    bench: str,
    reports: dict[str, dict[str, Any]],
    reader: ModelPrice,
    embed: ModelPrice,
    rag_ingest: Callable[[str], float] | None,
    by: str,
    keep: set[str] | None = None,
) -> list[dict[str, Any]]:
    """One row per (history, group, memory arm) with its break-even counts.
    reports maps arm letter to a loaded result file; "A" is required. keep, when given, is the set
    of history ids the loader yields: older result files can hold histories no longer in the
    benchmark set (the MemoryAgentBench length-ablation rows), and those are dropped."""
    fc = [c for c in _ok(reports["A"]["cases"]) if keep is None or c["history_id"] in keep]
    history_tokens = {c["history_id"]: c["tokens"]["context"] for c in fc}
    rows: list[dict[str, Any]] = []
    for arm in MEMORY_ARMS:
        if arm not in reports:
            continue
        mem = {c["item_id"]: c for c in _ok(reports[arm]["cases"])}
        recorded = ingest_by_history(reports[arm])
        groups: dict[tuple[str, str], tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
        for c in fc:
            if c["item_id"] not in mem:
                continue  # paired questions only
            hid = c["history_id"]
            group = bucket_label(history_tokens[hid]) if by == "bucket" else str(c["category"])
            pair = groups.setdefault((hid, group), ([], []))
            pair[0].append(c)
            pair[1].append(mem[c["item_id"]])
        for (hid, group), (fc_cases, mem_cases) in sorted(groups.items()):
            if hid in recorded:
                ingest = float(recorded[hid]["usd"])
            elif arm == "C" and rag_ingest is not None:
                ingest = rag_ingest(hid)
            else:
                continue  # no measured ingestion for this history
            rows.append({
                "bench": bench, "arm": arm, "history_id": hid, "group": group,
                "history_tokens": history_tokens[hid], "questions": len(fc_cases), "ingest_usd": ingest,
                **break_even(reader, embed, history_tokens[hid], ingest, fc_cases, mem_cases),
            })
    return rows


def summarize(rows: Sequence[dict[str, Any]], reps: int = 2000, seed: int = 0) -> list[dict[str, Any]]:
    """One table row per group: history count, benchmark mix, median history length, and the
    median break-even (with 95% interval) for each memory arm and mode."""
    groups = sorted({r["group"] for r in rows}, key=lambda g: (_bucket_key(g) if "K-" in g else 0, g))
    table: list[dict[str, Any]] = []
    for g in groups:
        here = [r for r in rows if r["group"] == g]
        histories = {(r["bench"], r["history_id"]): r["history_tokens"] for r in here}
        row: dict[str, Any] = {
            "group": g,
            "histories": len(histories),
            "benchmarks": dict(Counter(b for b, _ in histories)),
            "median_history_tokens": statistics.median(histories.values()),
        }
        for arm in MEMORY_ARMS:
            mine = [r for r in here if r["arm"] == arm]
            for mode in MODES:
                vals = [r[mode] for r in mine]
                row[f"{arm}_{mode}"] = _median_ci(vals, reps, seed) if vals else None
        table.append(row)
    return table


def _num(x: float) -> str:
    return "none" if math.isinf(x) else f"{x:,.0f}"


def _cell(v: tuple[float, float, float] | None) -> str:
    if v is None:
        return "not run"
    return f"{_num(v[0])} [{_num(v[1])}, {_num(v[2])}]"


def to_markdown(table: Sequence[dict[str, Any]], by: str) -> str:
    head = "History length" if by == "bucket" else "Question category"
    cols = [f"{name}, {'warm cache' if m == 'warm' else 'no cache'}" for name in MEMORY_ARMS.values() for m in MODES]
    lines = [
        f"| {head} | Histories | Benchmarks | Median tokens | " + " | ".join(cols) + " |",
        "| --- | --- | --- | --- | " + " | ".join("---" for _ in cols) + " |",
    ]
    for r in table:
        mix = ", ".join(f"{b} {n}" for b, n in sorted(r["benchmarks"].items()))
        cells = [_cell(r[f"{a}_{m}"]) for a in MEMORY_ARMS for m in MODES]
        lead = f"| {r['group']} | {r['histories']} | {mix} | {r['median_history_tokens']:,.0f} | "
        lines.append(lead + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Break-even = questions per history after which the memory arm is cheaper than full context "
        "(median over histories, 95% bootstrap interval). \"none\": no crossover."
    )
    return "\n".join(lines)


def load_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(Path(path).read_text())
    return report
