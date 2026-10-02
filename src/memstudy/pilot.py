"""Pilot tooling: sampling, judge stability check, containment diagnostic, cost projection.

The judge (GPT-5 nano) is the sole grader. There is no human grading and no second grader. The
stability check reruns the judge on 100 answers (50 from arm A, 50 from arm B) and reports its
flip rate. A deterministic containment test on short-answer questions is reported as a
diagnostic only. Only rerun_judge calls a model (the study judge).
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from memstudy.budget import ModelPrice
from memstudy.costmodel import QueryShape, full_context_cost, memory_cost
from memstudy.judge import Judge
from memstudy.schema import Item
from memstudy.store import ResultStore

SHORT_ANSWER_WORDS = 5
QUESTION_TOKENS = 60  # approximate size of the question block, negligible next to the context


def stratified_sample(items: list[Item], n: int, rng: random.Random) -> list[Item]:
    """Round-robin across categories after a seeded shuffle, so strata stay balanced."""
    pools: dict[str, list[Item]] = defaultdict(list)
    for item in sorted(items, key=lambda i: i.item_id):
        pools[item.category].append(item)
    for pool in pools.values():
        rng.shuffle(pool)
    order = sorted(pools)
    picked: list[Item] = []
    while len(picked) < n and any(pools.values()):
        for cat in order:
            if pools[cat] and len(picked) < n:
                picked.append(pools[cat].pop())
    return sorted(picked, key=lambda i: i.item_id)


def select_pilot(
    items_by_bench: dict[str, list[Item]], cfg: dict[str, Any]
) -> dict[str, list[Item]]:
    """Deterministic pilot sample. LoCoMo is limited to a few conversations to bound ingestion
    cost for the memory arms; LongMemEval-S is stratified by question type."""
    pilot = cfg["pilot"]
    rng = random.Random(cfg["seed"])
    locomo = [i for i in items_by_bench["locomo"] if i.primary]
    convs = sorted({i.history_id for i in locomo})
    chosen = set(rng.sample(convs, pilot["locomo_conversations"]))
    locomo_pool = [i for i in locomo if i.history_id in chosen]
    return {
        "locomo": stratified_sample(locomo_pool, pilot["locomo_questions"], rng),
        "longmemeval": stratified_sample(
            items_by_bench["longmemeval"], pilot["longmemeval_questions"], rng
        ),
    }


def select_flip_sample(
    store: ResultStore, per_arm: int, seed: int, arms: tuple[str, ...] = ("A", "B")
) -> list[dict[str, Any]]:
    """Graded records for the judge stability check, stratified by benchmark and category."""
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    for arm in arms:
        graded = [
            r
            for bench in ("locomo", "longmemeval")
            for r in store.read_items(arm, bench)
            if r.get("correct") is not None
        ]
        strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in sorted(graded, key=lambda r: (r["bench"], r["item_id"])):
            strata[(r["bench"], r["category"])].append(r)
        for pool in strata.values():
            rng.shuffle(pool)
        picked: list[dict[str, Any]] = []
        keys = sorted(strata)
        while len(picked) < min(per_arm, len(graded)):
            for key in keys:
                if strata[key] and len(picked) < per_arm:
                    picked.append(strata[key].pop())
        sample.extend(picked)
    return sample


def rerun_judge(
    judge: Judge, records: list[dict[str, Any]], out_path: Path, stage: str
) -> dict[str, Any]:
    """Grade the same answers again and report how often the verdict changes between runs.
    Refuses before any paid call if the report already exists."""
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists")
    flips: dict[str, list[bool]] = defaultdict(list)
    second: dict[str, bool | None] = {}
    for r in records:
        item = Item(
            item_id=r["item_id"],
            bench=r["bench"],
            history_id=r["history_id"],
            category=r["category"],
            question=r["question"],
            gold=r["gold"],
            meta={"abstention": bool(r.get("abstention", False))},
        )
        result = judge.grade(
            stage=stage, tag={"arm": r["arm"], "item": r["item_id"]}, item=item, model_answer=r["model_answer"]
        )
        second[f"{r['arm']}/{r['item_id']}"] = result.correct
        flips[r["arm"]].append(result.correct != r["correct"])
    total = sum(len(v) for v in flips.values())
    report = {
        "n": total,
        "flips": sum(sum(v) for v in flips.values()),
        "flip_rate": sum(sum(v) for v in flips.values()) / total if total else 0.0,
        "flip_rate_by_arm": {arm: sum(v) / len(v) for arm, v in flips.items()},
        "rerun": second,
    }
    with out_path.open("x") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return report


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def containment_diagnostic(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Agreement between the judge and a deterministic containment test, on short-answer,
    non-abstention questions. Diagnostic only: containment is not a grader."""
    agree: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        gold = _normalize(r["gold"])
        if r.get("abstention") or r.get("correct") is None or not gold:
            continue
        if len(gold.split()) > SHORT_ANSWER_WORDS:
            continue
        contained = gold in _normalize(r["model_answer"])
        agree[r["arm"]].append(contained == r["correct"])
    n = sum(len(v) for v in agree.values())
    return {
        "n": n,
        "agreement": sum(sum(v) for v in agree.values()) / n if n else None,
        "agreement_by_arm": {arm: sum(v) / len(v) for arm, v in agree.items()},
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pilot_stats(
    store: ResultStore, arm: str, bench: str, history_tokens: dict[str, int]
) -> dict[str, Any]:
    """Measured per-query and ingestion figures for one arm on one benchmark."""
    records = store.read_items(arm, bench)
    ingests = store.read_ingest(arm, bench)
    inp = [r["reader"]["usage"]["input_tokens"] for r in records]
    out = [r["reader"]["usage"]["output_tokens"] for r in records]
    cached = [r["reader"]["usage"]["cached_tokens"] for r in records]
    written = [r["reader"]["usage"]["cache_write_tokens"] for r in records]
    return {
        "n": len(records),
        "tokens_per_query": {
            "input": _mean(inp),
            "cached": _mean(cached),
            "cache_write": _mean(written),
            "output": _mean(out),
        },
        "cache_hit_rate": (sum(cached) / sum(inp)) if sum(inp) else 0.0,
        "reader_usd_per_query": _mean([r["reader"]["cost_usd"] for r in records]),
        "retrieval_usd_per_query": _mean([r["retrieval"]["cost"]["usd"] for r in records]),
        "judge_usd_per_query": _mean([r["judge"]["cost_usd"] for r in records if "judge" in r]),
        "latency_s_mean": _mean([r["latency_s"]["query_total"] for r in records]),
        "context_tokens_mean": _mean([r["context_tokens_counted"] for r in records]),
        "ingest": {
            "histories": len(ingests),
            "usd_total": sum(i["cost"]["usd"] for i in ingests),
            "seconds_total": sum(i["seconds"] for i in ingests),
            "usd_per_history": _mean([i["cost"]["usd"] for i in ingests]),
            "usd_per_token": [i["cost"]["usd"] / history_tokens[i["history_id"]] for i in ingests],
        },
    }


def project_full_cost(
    price: ModelPrice,
    arm: str,
    stats: dict[str, Any],
    full: list[tuple[int, int]],
) -> dict[str, float]:
    """Projected full-benchmark reader-side cost with a range. full lists (history_tokens,
    queries) for every history of the benchmark. Judge cost is added separately by the caller.

    Arm A: low assumes the cache stays warm (one write, then reads), high assumes no cache hits.
    Arms B and C: ingestion cost is scaled by history tokens using the measured USD per token;
    low, mid, and high are the cheapest, mean, and dearest pilot history.
    """
    shape = QueryShape(
        question_tokens=QUESTION_TOKENS,
        answer_tokens=int(stats["tokens_per_query"]["output"]) or 20,
    )
    if arm == "A":
        low = sum(full_context_cost(price, h, q, shape, True) for h, q in full)
        high = sum(full_context_cost(price, h, q, shape, False) for h, q in full)
        return {"low": low, "mid": (low + high) / 2, "high": high}
    ctx = int(stats["context_tokens_mean"])
    per_query_extra = stats["retrieval_usd_per_query"]
    per_token: list[float] = stats["ingest"]["usd_per_token"]

    def total(rate: float) -> float:
        return sum(memory_cost(price, rate * h, q, ctx, shape, per_query_extra) for h, q in full)

    return {
        "low": total(min(per_token)),
        "mid": total(_mean(per_token)),
        "high": total(max(per_token)),
    }
