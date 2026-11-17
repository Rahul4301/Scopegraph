"""RQ4 break-even table: offline, from result-file shaped dicts."""

from __future__ import annotations

import json
import math

from memstudy import rq4
from memstudy.budget import ModelPrice
from memstudy.execute import _merged_ingest

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
        "question": "When did Caroline go to the LGBTQ support group?",
        "tokens": {"context": context, "input": context + 50, "output": 10},
    }


# --- solve: integer semantics on piecewise-linear costs ------------------------------------


def test_forward_crossover_uses_strict_inequality_at_exact_ties():
    # fc = 3Q, mem = 4 + Q: equal at Q = 2, memory strictly cheaper from Q = 3.
    assert rq4.solve(lambda q: 3.0 * q, lambda q: 4.0 + q) == (3.0, None)


def test_memory_cheaper_only_for_the_first_questions_is_reported_not_hidden():
    # fc = 15 + 5Q, mem = 6Q: memory cheaper for Q < 15 only.
    assert rq4.solve(lambda q: 15.0 + 5 * q, lambda q: 6.0 * q) == (math.inf, 14)


def test_equal_slopes_with_memory_cheaper_everywhere_break_even_at_once():
    assert rq4.solve(lambda q: 3.0 + q, lambda q: 0.5 + q) == (1.0, None)


def test_never_cheaper():
    assert rq4.solve(lambda q: float(q), lambda q: 1.0 + 2 * q) == (math.inf, None)


# --- break_even on realistic token counts ---------------------------------------------------


def test_long_history_breaks_even_at_once():
    q = rq4.break_even(READER, EMBED, 100_000, 0.002, [case("h", 0, 100_000)], [case("h", 0, 7_000)])
    assert (q["warm"], q["nocache"]) == (1.0, 1.0)


def test_short_history_against_a_warm_cache_is_cheaper_only_for_a_few_questions():
    q = rq4.break_even(READER, EMBED, 20_000, 0.0004, [case("h", 0, 20_000)], [case("h", 0, 7_000)])
    assert math.isinf(q["warm"]) and q["warm_until"] is not None and 1 <= q["warm_until"] < 10
    assert q["nocache"] == 1.0


def test_mem0_ingestion_cost_pushes_the_warm_break_even_out():
    q = rq4.break_even(READER, EMBED, 100_000, 0.14, [case("h", 0, 100_000)], [case("h", 0, 7_000)])
    assert 300 < q["warm"] < 600 and 10 < q["nocache"] < 20


def test_embedding_tokens_are_counted_in_cl100k():
    assert rq4.embed_tokens("hello world") == 2


# --- rows, ingest records, summary ----------------------------------------------------------


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
    assert table[0]["C_n"] == 1 and table[0]["C_warm"][1] is None  # n < 10: point only
    md = rq4.to_markdown(table, "bucket")
    assert "(1) none; Q≤" in md and "only in 1/1" in md


def test_a_zero_rag_ingest_record_from_the_embedding_cache_is_recomputed():
    a = {"cases": [case("h1", 0, 150_000)]}
    c = {"cases": [case("h1", 0, 7_000)], "summary": {"ingest": {"by_history": {"h1": {"usd": 0.0}}}}}
    rows = rq4.per_history_rows("b", {"A": a, "C": c}, READER, EMBED, lambda h: 0.003, "bucket")
    assert rows[0]["ingest_usd"] == 0.003


def test_a_memory_arm_without_ingest_records_is_skipped_and_reported():
    a = {"cases": [case("h1", 0, 150_000)]}
    b = {"cases": [case("h1", 0, 7_000)], "summary": {}}
    c = {"cases": [case("h1", 0, 7_000)], "summary": {}}
    notes: dict = {}
    rows = rq4.per_history_rows(
        "bench", {"A": a, "B": b, "C": c}, READER, EMBED, lambda h: 0.001, "category", notes=notes
    )
    assert {r["arm"] for r in rows} == {"C"} and rows[0]["group"] == "x"
    assert notes["skipped_no_ingest"] == {"B": {"bench": 1}}
    table = rq4.summarize(rows, reps=10)
    assert table[0]["B_warm"] is None
    md = rq4.to_markdown(table, "category", notes["supplied"])
    assert "no ingest data" in md and "every question on the history were of that category" in md


def test_histories_outside_the_loaded_set_are_dropped():
    a = {"cases": [case("keep", 0, 150_000), case("ablation", 0, 150_000)]}
    c = {"cases": [case("keep", 0, 7_000), case("ablation", 0, 7_000)], "summary": {}}
    notes: dict = {}
    rows = rq4.per_history_rows(
        "b", {"A": a, "C": c}, READER, EMBED, lambda h: 0.001, "bucket", keep={"keep"}, notes=notes
    )
    assert [r["history_id"] for r in rows] == ["keep"] and notes["dropped_histories"] == {"b": 1}


def test_interval_appears_from_ten_histories_and_output_is_standard_json():
    a = {"cases": [case(f"h{i}", 0, 150_000) for i in range(12)]}
    c = {"cases": [case(f"h{i}", 0, 7_000) for i in range(12)], "summary": {}}
    rows = rq4.per_history_rows("b", {"A": a, "C": c}, READER, EMBED, lambda h: 0.001, "bucket")
    table = rq4.summarize(rows, reps=50)
    assert table[0]["C_warm"][1] is not None
    rows[0]["warm"] = math.inf
    json.dumps(rq4.jsonable({"rows": rows, "table": table}), allow_nan=False)


def test_merge_keeps_the_paid_ingest_record_over_a_cached_zero():
    paid = {"summary": {"ingest": {"by_history": {"h": {"usd": 0.004}}}}}
    cached = {"summary": {"ingest": {"by_history": {"h": {"usd": 0.0}}}}}
    assert _merged_ingest([cached, paid])["h"]["usd"] == 0.004
