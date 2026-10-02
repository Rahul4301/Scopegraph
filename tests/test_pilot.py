import csv
import json
import random

import pytest
from conftest import make_item

from memstudy.pilot import (
    baseline_check,
    evaluate_judge_check,
    export_judge_check,
    project_full_cost,
    select_pilot,
    stratified_sample,
)
from memstudy.store import ResultStore


def _items(bench, n_hist, per_hist, cats):
    out = []
    for h in range(n_hist):
        for n in range(per_hist):
            item = make_item(f"{bench}{h}", n, bench)
            out.append(item.model_copy(update={"category": cats[n % len(cats)]}))
    return out


def test_stratified_sample_is_balanced_and_deterministic():
    items = _items("locomo", 3, 40, ["c1", "c2", "c3", "c4"])
    a = stratified_sample(items, 30, random.Random(0))
    b = stratified_sample(items, 30, random.Random(0))
    assert [i.item_id for i in a] == [i.item_id for i in b]
    counts = {c: sum(i.category == c for i in a) for c in ("c1", "c2", "c3", "c4")}
    assert max(counts.values()) - min(counts.values()) <= 1 and sum(counts.values()) == 30


def test_pilot_selection_matches_the_pregistered_sizes(cfg):
    by_bench = {
        "locomo": _items("locomo", 10, 60, ["cat1", "cat2", "cat3", "cat4"]),
        "longmemeval": _items("lme", 100, 1, ["a", "b", "c", "d", "e", "f"]),
    }
    pilot = select_pilot(by_bench, cfg)
    assert len(pilot["locomo"]) == 30 and len(pilot["longmemeval"]) == 20
    assert len({i.history_id for i in pilot["locomo"]}) <= 3
    assert pilot == select_pilot(by_bench, cfg)


def _record(arm, bench, n, verdict, category="cat1"):
    return {
        "arm": arm, "bench": bench, "item_id": f"{bench}-{n:03d}", "category": category,
        "question": f"q{n}", "gold": "alpha", "model_answer": f"{arm}-answer-{n}",
        "judge": {"raw": "x"}, "correct": verdict,
    }


@pytest.fixture
def exported(tmp_path):
    store = ResultStore(tmp_path / "res")
    for arm in ("A", "B"):
        for n in range(8):
            store.write_item(arm, "locomo", f"locomo-{n:03d}", _record(arm, "locomo", n, n % 2 == 0))
    out = tmp_path / "check"
    export_judge_check(store, out, n_per_arm=6, seed=0)
    return out


def test_labeling_copy_hides_the_judge_verdict_and_the_arm(exported):
    text = (exported / "labeling.csv").read_text()
    rows = list(csv.DictReader(text.splitlines()))
    assert len(rows) == 12
    assert set(rows[0]) == {
        "check_id", "bench", "category", "question", "gold", "model_answer", "human_verdict"
    }
    assert all(r["human_verdict"] == "" for r in rows)
    assert "CORRECT" not in text and "nano_verdict" not in text
    key = json.loads((exported / "answer_key.json").read_text())
    assert {v["arm"] for v in key.values()} == {"A", "B"}
    assert sum(v["arm"] == "A" for v in key.values()) == 6


def test_export_refuses_to_overwrite(exported, tmp_path):
    store = ResultStore(tmp_path / "res")
    with pytest.raises(FileExistsError):
        export_judge_check(store, exported, n_per_arm=6, seed=0)


def _label(exported, tmp_path, flip_arm=None, flip_n=0):
    key = json.loads((exported / "answer_key.json").read_text())
    rows = list(csv.DictReader((exported / "labeling.csv").open()))
    flipped = 0
    for r in rows:
        verdict = key[r["check_id"]]["nano_verdict"]
        if key[r["check_id"]]["arm"] == flip_arm and flipped < flip_n:
            verdict = "INCORRECT" if verdict == "CORRECT" else "CORRECT"
            flipped += 1
        r["human_verdict"] = verdict
    path = tmp_path / "labeled.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_judge_check_passes_with_full_agreement(exported, tmp_path):
    result = evaluate_judge_check(_label(exported, tmp_path), exported / "answer_key.json", 0.95, 0.02)
    assert result["pass"] and result["agreement"] == 1.0


def test_judge_check_fails_below_95_percent(exported, tmp_path):
    labeled = _label(exported, tmp_path, flip_arm="A", flip_n=2)
    result = evaluate_judge_check(labeled, exported / "answer_key.json", 0.95, 0.02)
    assert not result["pass"] and result["agreement"] < 0.95


def test_judge_check_fails_when_arm_disagreement_gap_exceeds_two_points(exported, tmp_path):
    labeled = _label(exported, tmp_path, flip_arm="A", flip_n=1)
    result = evaluate_judge_check(labeled, exported / "answer_key.json", 0.80, 0.02)
    assert result["disagreement_gap"] > 0.02 and not result["pass"]


def test_baseline_gate_stops_at_extremes():
    assert baseline_check(0, 5, 0.10, 0.90)["stop"]
    assert baseline_check(5, 5, 0.10, 0.90)["stop"]
    assert not baseline_check(2, 5, 0.10, 0.90)["stop"]


def test_projection_has_an_ordered_range(prices):
    stats = {
        "tokens_per_query": {"output": 8},
        "context_tokens_mean": 7000,
        "retrieval_usd_per_query": 1e-7,
        "ingest": {"usd_per_token": [1e-7, 2e-7, 3e-7]},
    }
    full = [(100_000, 50), (120_000, 40)]
    a = project_full_cost(prices["gpt-6-luna"], "A", stats, full)
    b = project_full_cost(prices["gpt-6-luna"], "B", stats, full)
    assert a["low"] < a["mid"] < a["high"]
    assert b["low"] < b["mid"] < b["high"]
