"""RQ4: after how many questions per history are RAG (C) and Mem0 (B) cheaper than full context (A)?

Free and offline: reads finished result files, never calls an API. For each history it takes the
measured inputs of the closed-form cost model (costmodel.py) and finds, over whole question counts
Q >= 1, where the memory arm is strictly cheaper than full context, once with the full-context
cache kept warm between questions and once with no cache:

- history tokens H: arm A's context tokens for that history;
- non-history prompt and answer tokens: per-arm means over the paired questions, each priced at the
  tier of that arm's own prompt (full-context prompts above 272K tokens pay the long-context rate);
- memory read: the memory arm's mean retrieved context tokens K, read uncached, plus embedding the
  question text (cl100k tokens, as the embedding model bills them);
- ingestion I: the memory arm's measured per-history ingestion cost. For RAG it is recomputed from
  its chunks (cl100k tokens x the embedding price) when no record exists or the record is $0
  because the embeddings came from the local cache.

Warm cache mirrors arm A: a history asked once pays the plain input rate; from the second question
the first one writes the cache and later ones read it (every question within the cache lifetime).

Each history gets a break-even Q* (memory cheaper for every Q >= Q*, inf when that never happens)
and, when memory is cheaper only for the first few questions, the last such Q. Rows group histories
into 100K-token buckets (`by="bucket"`) or questions by category (`by="category"`: the break-even if
every question on the history were of that category; ingestion is still charged once per history).
Cells give the low median over histories and, from 10 histories up, a 95% percentile bootstrap
interval over histories (10,000 resamples, seed 0). The judge is excluded from both arms.
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
from memstudy.costmodel import QueryShape, full_context_cost, memory_cost
from memstudy.execute import ingest_by_history

BUCKET = 100_000
MEMORY_ARMS = {"C": "RAG", "B": "Mem0"}
MODES = ("warm", "nocache")
MIN_FOR_INTERVAL = 10


def _ok(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in cases if c.get("status") == "ok" and c.get("tokens")]


def bucket_label(tokens: int) -> str:
    lo = (tokens // BUCKET) * BUCKET
    return f"{lo // 1000}K-{(lo + BUCKET) // 1000}K"


def _bucket_key(label: str) -> int:
    return int(label.split("K", 1)[0])


def _shape(cases: Sequence[dict[str, Any]]) -> QueryShape:
    """Mean non-history prompt tokens (instructions and question) and answer tokens of one arm."""
    question = statistics.fmean(c["tokens"]["input"] - c["tokens"]["context"] for c in cases)
    answer = statistics.fmean(c["tokens"]["output"] for c in cases)
    return QueryShape(question_tokens=round(question), answer_tokens=round(answer))


def embed_tokens(text: str) -> int:
    """Tokens as text-embedding-3-small bills them (cl100k_base, not the reader's o200k_base)."""
    import tiktoken

    return len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=()))


def solve(fc: Callable[[int], float], mem: Callable[[int], float]) -> tuple[float, int | None]:
    """Where mem(Q) < fc(Q) over integers Q >= 1. fc and mem are linear from Q = 2 on (fc may
    differ at Q = 1). Returns (break_even, cheaper_until): break_even is the smallest Q such that
    memory is cheaper at every Q' >= Q (inf if none); cheaper_until is the last Q at which memory
    is cheaper when its advantage does not last (None otherwise)."""
    q1 = mem(1) < fc(1)
    d2, d3 = mem(2) - fc(2), mem(3) - fc(3)
    beta = d3 - d2
    alpha = d2 - 2 * beta  # mem(Q) - fc(Q) = alpha + beta*Q for Q >= 2
    eps = 1e-15
    if beta < -eps:  # memory gets relatively cheaper: cheaper for Q > -alpha/beta
        start = max(2, math.floor(-alpha / beta) + 1)
        if mem(start) - fc(start) >= 0:  # guard float error at the boundary
            start += 1
        return (1.0 if start == 2 and q1 else float(start)), None
    if beta > eps:  # memory gets relatively dearer: cheaper only for Q < -alpha/beta
        last = math.ceil(-alpha / beta) - 1
        while last >= 2 and mem(last) - fc(last) >= 0:
            last -= 1
        if last >= 2:
            return math.inf, last  # cheaper over [2, last] (and at 1 when q1)
        return math.inf, (1 if q1 else None)
    if alpha < 0:  # equal slopes, memory cheaper from Q = 2 on
        return (1.0 if q1 else 2.0), None
    return math.inf, (1 if q1 else None)


def break_even(
    reader: ModelPrice,
    embed: ModelPrice,
    history_tokens: int,
    ingest_usd: float,
    fc_cases: Sequence[dict[str, Any]],
    mem_cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Break-even for one history (or one category of its questions), per mode."""
    fc_shape, mem_shape = _shape(fc_cases), _shape(mem_cases)
    context = round(statistics.fmean(c["tokens"]["context"] for c in mem_cases))
    retrieval = statistics.fmean(embed_tokens(str(c.get("question", ""))) for c in mem_cases) * embed.input / 1e6

    def mem(q: int) -> float:
        return memory_cost(reader, ingest_usd, q, context, mem_shape, retrieval)

    out: dict[str, Any] = {}
    for mode in MODES:
        def fc(q: int, mode: str = mode) -> float:
            # Arm A sets the cache breakpoint only when a history gets two or more questions.
            return full_context_cost(reader, history_tokens, q, fc_shape, cached=(mode == "warm" and q >= 2))

        out[mode], out[f"{mode}_until"] = solve(fc, mem)
    return out


def rag_ingest_usd(history: Any, chunk_tokens: int, embed: ModelPrice) -> float:
    """Arm C ingestion recomputed from its chunks: every chunk is embedded once, billed in cl100k
    tokens (matches the API's recorded usage exactly on LoCoMo)."""
    from memstudy.arms.rag import chunk_history

    return sum(embed_tokens(c) for c in chunk_history(history, chunk_tokens)) * embed.input / 1e6


def rag_ingest_from(histories: dict[str, Any], chunk_tokens: int, embed: ModelPrice) -> Callable[[str], float]:
    """Arm C ingestion cost by history id, recomputed from the loaded histories."""
    return lambda history_id: rag_ingest_usd(histories[history_id], chunk_tokens, embed)


def _median_ci(values: Sequence[float], reps: int, seed: int) -> tuple[float, float | None, float | None]:
    """Low median (a whole question count) and, from MIN_FOR_INTERVAL values up, a percentile
    bootstrap interval. inf sorts last, so a median of inf means at least half never break even."""
    point = statistics.median_low(values)
    if len(values) < MIN_FOR_INTERVAL:
        return point, None, None
    rng = random.Random(seed)
    n = len(values)
    meds = sorted(statistics.median_low(rng.choices(values, k=n)) for _ in range(reps))
    return point, meds[int(0.025 * reps)], meds[min(reps - 1, int(0.975 * reps))]


def per_history_rows(
    bench: str,
    reports: dict[str, dict[str, Any]],
    reader: ModelPrice,
    embed: ModelPrice,
    rag_ingest: Callable[[str], float] | None,
    by: str,
    keep: set[str] | None = None,
    notes: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """One row per (history, group, memory arm) with its break-even counts.
    reports maps arm letter to a loaded result file; "A" is required. keep, when given, is the set
    of history ids the loader yields: older result files can hold histories no longer in the
    benchmark set (the MemoryAgentBench length-ablation rows, as-of checkpoints), and those are
    dropped. notes, when given, collects what was dropped or skipped, for the caller to report."""
    notes = notes if notes is not None else {}
    all_fc = _ok(reports["A"]["cases"])
    fc = [c for c in all_fc if keep is None or c["history_id"] in keep]
    dropped = {c["history_id"] for c in all_fc} - {c["history_id"] for c in fc}
    notes.setdefault("dropped_histories", {})[bench] = len(dropped)
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
        skipped: set[str] = set()
        for (hid, group), (fc_cases, mem_cases) in sorted(groups.items()):
            usd = float(recorded[hid]["usd"]) if hid in recorded else None
            if arm == "C" and (usd is None or usd == 0.0) and rag_ingest is not None:
                ingest = rag_ingest(hid)  # no record, or $0 because the vectors came from the cache
            elif usd is not None and usd > 0.0:
                ingest = usd
            else:
                skipped.add(hid)  # no measured ingestion for this history
                continue
            rows.append({
                "bench": bench, "arm": arm, "history_id": hid, "group": group,
                "history_tokens": history_tokens[hid], "questions": len(fc_cases), "ingest_usd": ingest,
                **break_even(reader, embed, history_tokens[hid], ingest, fc_cases, mem_cases),
            })
        if skipped:
            notes.setdefault("skipped_no_ingest", {}).setdefault(arm, {})[bench] = len(skipped)
        notes.setdefault("supplied", set()).add(arm)
    return rows


def summarize(rows: Sequence[dict[str, Any]], reps: int = 10_000, seed: int = 0) -> list[dict[str, Any]]:
    """One table row per group: history count, benchmark mix, median history length, and per memory
    arm its history count, the median break-even (with 95% interval) and, for each mode, how many
    histories have memory cheaper only for their first few questions (and the median last such Q)."""
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
            row[f"{arm}_n"] = len({(r["bench"], r["history_id"]) for r in mine})
            for mode in MODES:
                vals = [r[mode] for r in mine]
                row[f"{arm}_{mode}"] = _median_ci(vals, reps, seed) if vals else None
                early = [r[f"{mode}_until"] for r in mine if math.isinf(r[mode]) and r[f"{mode}_until"]]
                row[f"{arm}_{mode}_early"] = (len(early), statistics.median_low(early)) if early else None
        table.append(row)
    return table


def _num(x: float) -> str:
    return "none" if math.isinf(x) else f"{x:,.0f}"


def _cell(row: dict[str, Any], arm: str, mode: str, supplied: bool) -> str:
    v = row[f"{arm}_{mode}"]
    if v is None:
        return "no ingest data" if supplied else "not run"
    point, lo, hi = v
    text = _num(point) if lo is None or hi is None else f"{_num(point)} [{_num(lo)}, {_num(hi)}]"
    early = row[f"{arm}_{mode}_early"]
    if early:
        text += f"; Q≤{early[1]:,} only in {early[0]}/{row[f'{arm}_n']}"
    return text


def to_markdown(table: Sequence[dict[str, Any]], by: str, supplied: set[str] | None = None) -> str:
    supplied = supplied if supplied is not None else set(MEMORY_ARMS)
    head = "History length" if by == "bucket" else "Question category"
    modes = {"warm": "warm cache", "nocache": "no cache"}
    cols = [f"{name} (n), {modes[m]}" for name in MEMORY_ARMS.values() for m in MODES]
    lines = [
        f"| {head} | Histories | Benchmarks | Median tokens | " + " | ".join(cols) + " |",
        "| --- | --- | --- | --- | " + " | ".join("---" for _ in cols) + " |",
    ]
    for r in table:
        mix = ", ".join(f"{b} {n}" for b, n in sorted(r["benchmarks"].items()))
        cells = [
            (f"({r[f'{a}_n']}) " if r[f"{a}_n"] else "") + _cell(r, a, m, a in supplied)
            for a in MEMORY_ARMS for m in MODES
        ]
        lead = f"| {r['group']} | {r['histories']} | {mix} | {r['median_history_tokens']:,.0f} | "
        lines.append(lead + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Break-even Q* = fewest questions per history from which the memory arm stays strictly cheaper "
        "than full context (low median over histories; 95% bootstrap interval over histories when "
        f"n >= {MIN_FOR_INTERVAL}). \"none\": memory is not cheaper from any Q on. \"Q≤m only in k/n\": "
        "in k histories memory is cheaper only for the first m questions (median m). Warm cache: every "
        "question arrives within the cache lifetime. Judge cost excluded."
        + (" Category rows: break-even if every question on the history were of that category; "
           "ingestion is charged once per history." if by == "category" else "")
    )
    return "\n".join(lines)


def jsonable(value: Any) -> Any:
    """inf (no crossover) becomes None so --out is standard JSON."""
    if isinstance(value, float) and math.isinf(value):
        return None
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return value


def load_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(Path(path).read_text())
    return report
