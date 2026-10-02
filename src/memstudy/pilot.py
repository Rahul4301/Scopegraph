"""Phase 2 pilot tooling: sampling, judge hand-check, coding baseline gate, projection report.

Nothing here calls a model except rerun_judge, which uses the study judge (GPT-5 nano).
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from memstudy.budget import ModelPrice
from memstudy.costmodel import QueryShape, full_context_cost, memory_cost
from memstudy.judge import Judge
from memstudy.schema import Item
from memstudy.store import ResultStore


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
    """Deterministic pilot sample. LoCoMo is limited to a few conversations to bound Mem0
    ingestion cost; LongMemEval-S is stratified by question type."""
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


def _verdict_str(correct: bool | None) -> str:
    return {True: "CORRECT", False: "INCORRECT", None: "UNPARSEABLE"}[correct]


def export_judge_check(
    store: ResultStore, out_dir: Path, n_per_arm: int, seed: int, arms: tuple[str, ...] = ("A", "B")
) -> dict[str, int]:
    """Sample graded answers per arm, stratified by bench and category. Writes a labeling CSV
    with the judge verdict and the arm hidden, and a separate answer key. Refuses to overwrite."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for arm in arms:
        graded = [
            r
            for bench in ("locomo", "longmemeval")
            for r in store.read_items(arm, bench)
            if "judge" in r
        ]
        strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in sorted(graded, key=lambda r: (r["bench"], r["item_id"])):
            strata[(r["bench"], r["category"])].append(r)
        for pool in strata.values():
            rng.shuffle(pool)
        picked: list[dict[str, Any]] = []
        keys = sorted(strata)
        while len(picked) < min(n_per_arm, len(graded)):
            for key in keys:
                if strata[key] and len(picked) < n_per_arm:
                    picked.append(strata[key].pop())
        rows.extend(picked)
    rng.shuffle(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    labeling, key_file = out_dir / "labeling.csv", out_dir / "answer_key.json"
    if labeling.exists() or key_file.exists():
        raise FileExistsError(f"judge check already exported under {out_dir}")
    answer_key = {}
    with labeling.open("x", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["check_id", "bench", "category", "question", "gold", "model_answer", "human_verdict"]
        )
        for n, r in enumerate(rows):
            cid = f"c{n:03d}"
            writer.writerow(
                [cid, r["bench"], r["category"], r["question"], r["gold"], r["model_answer"], ""]
            )
            answer_key[cid] = {
                "arm": r["arm"],
                "bench": r["bench"],
                "item_id": r["item_id"],
                "category": r["category"],
                "abstention": bool(r.get("abstention", False)),
                "question": r["question"],
                "gold": r["gold"],
                "model_answer": r["model_answer"],
                "nano_verdict": _verdict_str(r["correct"]),
            }
    with key_file.open("x") as fh:
        json.dump(answer_key, fh, indent=2, sort_keys=True)
    return {"labeling_rows": len(rows)}


def rerun_judge(judge: Judge, key_path: Path, out_path: Path, stage: str) -> dict[str, Any]:
    """Run the judge again on the exported items and report its flip rate between runs."""
    key = json.loads(key_path.read_text())
    flips = 0
    second: dict[str, str] = {}
    for cid, row in sorted(key.items()):
        item = Item(
            item_id=row["item_id"],
            bench=row["bench"],
            history_id="-",
            category=row["category"],
            question=row["question"],
            gold=row["gold"],
            meta={"abstention": row["abstention"]},
        )
        result = judge.grade(
            stage=stage, tag={"arm": row["arm"], "item": cid}, item=item, model_answer=row["model_answer"]
        )
        second[cid] = _verdict_str(result.correct)
        flips += second[cid] != row["nano_verdict"]
    report = {"n": len(key), "flips": flips, "flip_rate": flips / len(key), "rerun": second}
    with out_path.open("x") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return report


def _parse_human(value: str) -> str:
    v = value.strip().lower()
    if v in {"correct", "c", "1", "y", "yes", "true"}:
        return "CORRECT"
    if v in {"incorrect", "i", "0", "n", "no", "false"}:
        return "INCORRECT"
    raise ValueError(f"unrecognised human verdict: {value!r}")


def evaluate_judge_check(
    labeled_csv: Path, key_path: Path, agreement_min: float, gap_max: float
) -> dict[str, Any]:
    """Pre-registered pass criteria: nano agrees with the human on at least agreement_min
    overall, and the two arms' disagreement rates are within gap_max of each other."""
    key = json.loads(key_path.read_text())
    per_arm: dict[str, list[bool]] = defaultdict(list)
    with labeled_csv.open(newline="") as fh:
        for row in csv.DictReader(fh):
            agree = _parse_human(row["human_verdict"]) == key[row["check_id"]]["nano_verdict"]
            per_arm[key[row["check_id"]]["arm"]].append(agree)
    total = sum(len(v) for v in per_arm.values())
    overall = sum(sum(v) for v in per_arm.values()) / total
    disagree = {arm: 1 - sum(v) / len(v) for arm, v in per_arm.items()}
    gap = max(disagree.values()) - min(disagree.values()) if len(disagree) > 1 else 0.0
    return {
        "n": total,
        "agreement": overall,
        "disagreement_by_arm": disagree,
        "disagreement_gap": gap,
        "pass": overall >= agreement_min and gap <= gap_max,
    }


def baseline_check(resolved: int, total: int, low: float, high: float) -> dict[str, Any]:
    """Memory effects cannot show when the memory-off solve rate is near 0 or near 100."""
    rate = resolved / total
    return {
        "resolved": resolved,
        "total": total,
        "rate": rate,
        "stop": rate <= low or rate >= high,
        "note": "stop and report: baseline at an extreme" if rate <= low or rate >= high else "ok",
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pilot_stats(
    store: ResultStore, arm: str, bench: str, history_tokens: dict[str, int]
) -> dict[str, Any]:
    """Measured per-query and ingestion figures for one arm on one benchmark."""
    records = store.read_items(arm, bench)
    ingests = store.read_ingest(arm, bench)
    n = len(records)
    inp = [r["reader"]["usage"]["input_tokens"] for r in records]
    out = [r["reader"]["usage"]["output_tokens"] for r in records]
    cached = [r["reader"]["usage"]["cached_tokens"] for r in records]
    written = [r["reader"]["usage"]["cache_write_tokens"] for r in records]
    return {
        "n": n,
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
            "usd_per_token": [
                i["cost"]["usd"] / history_tokens[i["history_id"]] for i in ingests
            ],
        },
    }


QUESTION_TOKENS = 60  # approximate size of the question block, negligible next to the context


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
        return sum(
            memory_cost(price, rate * h, q, ctx, shape, per_query_extra) for h, q in full
        )

    return {
        "low": total(min(per_token)),
        "mid": total(_mean(per_token)),
        "high": total(max(per_token)),
    }
