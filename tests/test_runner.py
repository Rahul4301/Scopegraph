import json

import pytest
from conftest import FakeClient, make_history, make_item, make_response

from memstudy.arms.base import ArmContext, IngestStats
from memstudy.arms.full_context import FullContextArm
from memstudy.budget import Budget, BudgetExceeded
from memstudy.config import config_hash
from memstudy.judge import Judge
from memstudy.llm import ModelCaller
from memstudy.metering import CostSink
from memstudy.reader import Reader
from memstudy.runner import finalize_run, run_chat
from memstudy.store import ResultStore


def handler(kw):
    if kw["model"] == "gpt-6-luna":
        return make_response(
            text="alpha", input_tokens=4000, cached=2048, written=1500, output=5, reasoning=0
        )
    return make_response(
        text='{"verdict": "CORRECT"}', model="gpt-5-nano-test", input_tokens=200, output=40, reasoning=30
    )


def build(prices, budget, cfg, fn=handler):
    client = FakeClient(fn)
    reader = Reader(ModelCaller(client, prices["gpt-6-luna"], budget), cfg)
    judge = Judge(ModelCaller(client, prices["gpt-5-nano"], budget), cfg)
    return client, reader, judge


def test_full_run_writes_raw_records_and_run_json(prices, budget, cfg, tmp_path):
    client, reader, judge = build(prices, budget, cfg)
    store = ResultStore(tmp_path / "res")
    history = make_history("h1")
    summary = run_chat(
        arm=FullContextArm(100_000, 0.05),
        items=[make_item("h1", 0), make_item("h1", 1)],
        histories={"h1": history},
        reader=reader,
        judge=judge,
        store=store,
        stage="pilot",
        run_id="run-1",
    )
    assert summary.completed == 2 and summary.errors == 0
    record = store.read_items("A", "locomo")[0]
    assert record["correct"] is True and record["model_answer"] == "alpha"
    assert record["reader"]["cached_tokens"] == 2048
    assert record["reader"]["usage"]["cache_write_tokens"] == 1500
    assert record["judge"]["reasoning_tokens"] == 30
    assert record["reader"]["model_returned"] == "gpt-6-luna-test-snapshot"

    run = finalize_run(
        store, run_id="run-1", arm="A", bench="locomo", stage="pilot",
        cfg_hash=config_hash(cfg), summary=summary,
    )
    on_disk = json.loads((store.root / "runs" / "run-1" / "run.json").read_text())
    assert on_disk["arm"] == "A" and on_disk["config_hash"] == config_hash(cfg)
    assert on_disk["reader_tokens"] == {
        "input": 8000, "cached": 4096, "cache_write": 3000, "output": 10, "reasoning": 0
    }
    assert on_disk["judge_tokens"]["reasoning"] == 60
    assert on_disk["cache_hit_rate"] == pytest.approx(4096 / 8000)
    assert on_disk["accuracy"] == 1.0 and run["n_items"] == 2
    assert "python" in on_disk["versions"]
    assert on_disk["cost_usd"]["reader"] > 0 and on_disk["cost_usd"]["judge"] > 0


def test_resume_skips_finished_items_and_makes_no_new_calls(prices, budget, cfg, tmp_path):
    client, reader, judge = build(prices, budget, cfg)
    store = ResultStore(tmp_path / "res")
    kwargs = dict(
        arm=FullContextArm(100_000, 0.05),
        items=[make_item("h1", 0), make_item("h1", 1)],
        histories={"h1": make_history("h1")},
        reader=reader, judge=judge, store=store, stage="pilot",
    )
    run_chat(run_id="r1", **kwargs)
    calls = len(client.responses.calls)
    again = run_chat(run_id="r2", **kwargs)
    assert again.completed == 0 and again.skipped_existing == 2
    assert len(client.responses.calls) == calls


def test_raw_results_are_never_overwritten(tmp_path):
    store = ResultStore(tmp_path / "res")
    store.write_item("A", "locomo", "x-0001", {"v": 1})
    with pytest.raises(FileExistsError):
        store.write_item("A", "locomo", "x-0001", {"v": 2})
    assert store.read_items("A", "locomo") == [{"v": 1}]
    store.write_ingest("B", "locomo", "h", {"v": 1})
    with pytest.raises(FileExistsError):
        store.write_ingest("B", "locomo", "h", {"v": 2})


def test_incomplete_reader_response_is_recorded_and_the_run_continues(prices, budget, cfg, tmp_path):
    state = {"n": 0}

    def flaky(kw):
        state["n"] += 1
        if kw["model"] == "gpt-6-luna" and state["n"] == 1:
            return make_response(status="incomplete")
        return handler(kw)

    _, reader, judge = build(prices, budget, cfg, flaky)
    store = ResultStore(tmp_path / "res")
    summary = run_chat(
        arm=FullContextArm(100_000, 0.05),
        items=[make_item("h1", 0), make_item("h1", 1)],
        histories={"h1": make_history("h1")},
        reader=reader, judge=judge, store=store, stage="pilot", run_id="r1",
    )
    assert summary.errors == 1 and summary.completed == 1
    assert len(store.read_errors("A", "locomo")) == 1


def test_budget_exhaustion_stops_the_run(prices, cfg, tmp_path):
    budget = Budget(tmp_path / "l.jsonl", stage_caps={"pilot": 0.0000001})
    _, reader, judge = build(prices, budget, cfg)
    with pytest.raises(BudgetExceeded):
        run_chat(
            arm=FullContextArm(100_000, 0.05),
            items=[make_item("h1", 0)],
            histories={"h1": make_history("h1")},
            reader=reader, judge=judge, store=ResultStore(tmp_path / "res"),
            stage="pilot", run_id="r1",
        )


class FakeMemoryArm:
    name = "B"

    def __init__(self):
        self.prepared = 0

    def prepare(self, history):
        self.prepared += 1
        return IngestStats(seconds=2.0, cost=CostSink(usd=0.5, calls=3), stored_units=7)

    def context(self, item, history):
        return ArmContext(text="- memory", context_tokens=3, cache_prefix=False, retrieved=["m1"])


def test_ingestion_is_recorded_once_per_history_and_not_repeated_on_resume(prices, budget, cfg, tmp_path):
    _, reader, judge = build(prices, budget, cfg)
    store = ResultStore(tmp_path / "res")
    arm = FakeMemoryArm()
    kwargs = dict(
        arm=arm, items=[make_item("h1", 0), make_item("h1", 1)],
        histories={"h1": make_history("h1")}, reader=reader, judge=judge,
        store=store, stage="pilot",
    )
    run_chat(run_id="r1", **kwargs)
    assert arm.prepared == 1
    ingest = store.read_ingest("B", "locomo")
    assert len(ingest) == 1 and ingest[0]["cost"]["usd"] == 0.5 and ingest[0]["stored_units"] == 7
    run_chat(run_id="r2", **kwargs)
    assert arm.prepared == 1  # every item was done, so the history is not re-ingested
