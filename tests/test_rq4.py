"""RQ4 break-even table: offline, from result-file shaped dicts."""

from __future__ import annotations

import math

from memstudy import rq4
from memstudy.budget import ModelPrice

READER = ModelPrice(
    name="reader", verified=True, input=0.10, cached_input=0.01, cache_write=0.125, output=0.50,
    long_context_threshold=272_000, long_context_multiplier={"input": 2.0, "cached_input": 2.0,
                                                             "cache_write": 2.0, "output": 1.5},
    context_window=1_050_000,
)
EMBED = ModelPrice(
    name="embed", verified=True, input=0.02, cached_input=0.02, cache_write=0.02, output=0.0,
    long_context_threshold=None, long_context_multiplier={}, context_window=None,
)


def case(hid: str, i: int, context: int, category: str = "x") -> dict:
    return {
        "item_id": f"{hid}-q{i}", "history_id": hid, "category": category, "status": "ok",
        "tokens": {"context": context, "input": context + 50, "output": 10},
    }


def test_bucket_labels_step_100k():
    assert rq4.bucket_label(19_771) == "0K-100K"
    assert rq4.bucket_label(100_000) == "100K-200K"
    assert rq4.bucket_label(534_000) == "500K-600K"


def test_long_history_breaks_even_at_once_and_short_one_never_against_a_warm_cache():
    long_ = rq4.break_even(READER, EMBED, 100_000, 0.002, [case("h", 0, 100_000)], [case("h", 0, 7_000)])
    assert long_ == {"warm": 1, "nocache": 1}
    short = rq4.break_even(READER, EMBED, 20_000, 0.002, [case("h", 0, 20_000)], [case("h", 0, 7_000)])
    assert math.isinf(short["warm"]) and short["nocache"] == 2  # $0.002 ingest takes two questions


def test_mem0_ingestion_cost_pushes_the_warm_break_even_out():
    q = rq4.break_even(READER, EMBED, 100_000, 0.14, [case("h", 0, 100_000)], [case("h", 0, 7_000)])
    assert 300 < q["warm"] < 600 and 10 < q["nocache"] < 20


def test_rows_pair_questions_use_recorded_ingest_and_summarize_by_bucket():
    a = {"cases": [case("h1", 0, 150_000), case("h1", 1, 150_000), case("h2", 0, 20_000)]}
    b = {
        "cases": [case("h1", 0, 7_000), case("h2", 0, 7_000)],  # h1-q1 unanswered by B: dropped
        "summary": {"ingest": {"by_history": {"h1": {"usd": 0.2}, "h2": {"usd": 0.03}}}},
    }
    c = {"cases": [case("h1", 0, 7_000), case("h1", 1, 7_000), case("h2", 0, 7_000)], "summary": {}}
    rows = rq4.per_history_rows("bench", {"A": a, "B": b, "C": c}, READER, EMBED, lambda h: 0.001, "bucket")
    by = {(r["arm"], r["history_id"]): r for r in rows}
    assert by[("B", "h1")]["questions"] == 1 and by[("C", "h1")]["questions"] == 2
    assert by[("B", "h1")]["ingest_usd"] == 0.2 and by[("C", "h2")]["ingest_usd"] == 0.001
    table = rq4.summarize(rows, reps=50)
    assert [r["group"] for r in table] == ["0K-100K", "100K-200K"]
    assert math.isinf(table[0]["C_warm"][0])
    assert "none [none, none]" in rq4.to_markdown(table, "bucket")


def test_a_memory_arm_without_ingest_records_is_skipped_but_rag_is_recomputed():
    a = {"cases": [case("h1", 0, 150_000)]}
    b = {"cases": [case("h1", 0, 7_000)], "summary": {}}
    c = {"cases": [case("h1", 0, 7_000)], "summary": {}}
    rows = rq4.per_history_rows("bench", {"A": a, "B": b, "C": c}, READER, EMBED, lambda h: 0.001, "category")
    assert {r["arm"] for r in rows} == {"C"} and rows[0]["group"] == "x"
    assert rq4.summarize(rows, reps=10)[0]["B_warm"] is None


def test_histories_outside_the_loaded_set_are_dropped():
    a = {"cases": [case("keep", 0, 150_000), case("ablation", 0, 150_000)]}
    c = {"cases": [case("keep", 0, 7_000), case("ablation", 0, 7_000)], "summary": {}}
    rows = rq4.per_history_rows("b", {"A": a, "C": c}, READER, EMBED, lambda h: 0.001, "bucket", keep={"keep"})
    assert [r["history_id"] for r in rows] == ["keep"]
