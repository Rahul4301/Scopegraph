import random

import pytest
from conftest import FakeClient, make_item, make_response

from memstudy.judge import Judge
from memstudy.llm import ModelCaller
from memstudy.pilot import (
    containment_diagnostic,
    project_full_cost,
    rerun_judge,
    select_flip_sample,
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


def _record(arm, bench, n, verdict, category="cat1", gold="alpha", answer=None):
    return {
        "arm": arm, "bench": bench, "item_id": f"{bench}-{n:03d}", "history_id": f"h{n % 3}",
        "category": category, "question": f"q{n}", "gold": gold,
        "model_answer": answer or f"{arm}-answer-{n}", "abstention": False,
        "judge": {"raw": "x"}, "correct": verdict,
    }


@pytest.fixture
def store(tmp_path):
    store = ResultStore(tmp_path / "res")
    for arm in ("A", "B"):
        for n in range(8):
            store.write_item(arm, "locomo", f"locomo-{n:03d}", _record(arm, "locomo", n, n % 2 == 0))
    return store


def test_flip_sample_takes_n_from_each_arm_deterministically(store):
    sample = select_flip_sample(store, per_arm=6, seed=0)
    assert [r["arm"] for r in sample].count("A") == 6 and [r["arm"] for r in sample].count("B") == 6
    assert sample == select_flip_sample(store, per_arm=6, seed=0)


def test_flip_sample_takes_everything_when_fewer_are_available(store):
    assert len(select_flip_sample(store, per_arm=50, seed=0)) == 16


def _judge(prices, budget, cfg, verdicts):
    queue = iter(verdicts)
    client = FakeClient(
        lambda kw: make_response(text=f'{{"verdict": "{next(queue)}"}}', model="gpt-5-nano", output=12)
    )
    return Judge(ModelCaller(client, prices["gpt-5-nano"], budget), cfg), client


def test_rerun_reports_the_flip_rate_overall_and_per_arm(store, prices, budget, cfg, tmp_path):
    records = select_flip_sample(store, per_arm=4, seed=0)
    # first-run verdicts are in the records; flip exactly two of the arm A answers
    verdicts, flipped = [], 0
    for r in records:
        again = "CORRECT" if r["correct"] else "INCORRECT"
        if r["arm"] == "A" and flipped < 2:
            again = "INCORRECT" if r["correct"] else "CORRECT"
            flipped += 1
        verdicts.append(again)
    judge, client = _judge(prices, budget, cfg, verdicts)
    report = rerun_judge(judge, records, tmp_path / "flip.json", "pilot")
    assert report["n"] == 8 and report["flips"] == 2 and report["flip_rate"] == 0.25
    assert report["flip_rate_by_arm"] == {"A": 0.5, "B": 0.0}
    assert len(client.responses.calls) == 8
    with pytest.raises(FileExistsError):
        rerun_judge(judge, records, tmp_path / "flip.json", "pilot")


def test_containment_diagnostic_covers_short_answers_only():
    records = [
        _record("A", "locomo", 0, True, gold="7 May 2023", answer="It was on 7 May, 2023."),
        _record("A", "locomo", 1, False, gold="Paris", answer="Rome"),
        _record("A", "locomo", 2, True, gold="Paris", answer="Rome"),  # judge and test disagree
        _record("B", "locomo", 3, True, gold="one two three four five six", answer="one two"),
        {**_record("B", "locomo", 4, True, gold="Not mentioned in the conversation."), "abstention": True},
    ]
    result = containment_diagnostic(records)
    assert result["n"] == 3 and result["agreement"] == pytest.approx(2 / 3)
    assert result["agreement_by_arm"] == {"A": pytest.approx(2 / 3)}


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
