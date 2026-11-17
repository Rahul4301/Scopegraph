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
from memstudy.runner import CACHE_KEY_MAX, cache_key, finalize_run, run_chat
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
    assert "judge_mem0" in record and record["judge_mem0"]["scheme"] == "mem0"
    assert on_disk["judge_tokens"]["reasoning"] == 120  # primary and Mem0 judge for two questions
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


def test_cache_key_stays_within_openai_limit_and_unique():
    short = cache_key("A", "locomo_conv-26")
    assert short == "A-locomo_conv-26"  # short keys are unchanged, so existing caches still hit
    a = cache_key("A", "memoryagentbench_long_range_understanding-infbench_sum_eng_shots2-r87")
    b = cache_key("A", "memoryagentbench_long_range_understanding-infbench_sum_eng_shots2-r88")
    assert len(a) <= CACHE_KEY_MAX and len(b) <= CACHE_KEY_MAX
    assert a != b
    assert a == cache_key("A", "memoryagentbench_long_range_understanding-infbench_sum_eng_shots2-r87")


def test_a_history_that_fails_to_store_is_recorded_and_the_run_continues(prices, budget, cfg, tmp_path):
    import openai

    class FailingArm(FullContextArm):
        def prepare(self, history):
            if history.history_id == "h1":
                raise openai.APIError("embedding rejected", request=None, body=None)
            return super().prepare(history)

    _, reader, judge = build(prices, budget, cfg)
    store = ResultStore(tmp_path)
    summary = run_chat(
        arm=FailingArm(100_000, 0.05), items=[make_item("h1", 0), make_item("h2", 0)],
        histories={"h1": make_history("h1"), "h2": make_history("h2")},
        reader=reader, judge=judge, store=store, stage="pilot", run_id="r",
    )
    assert summary.errors == 1 and summary.completed == 1
    assert store.read_errors("A", "locomo")[0]["error"] == "ingest_failed: APIError"


def test_embedding_and_mem0_calls_retry_a_dropped_connection(prices, budget):
    from types import SimpleNamespace as NS

    import openai

    from memstudy.metering import Meter
    from memstudy.schema import Usage

    meter = Meter(budget, prices, "pilot", CostSink(), {})
    meter._sleep = lambda s: None
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise openai.APIConnectionError(request=None)
        return NS(usage=NS(prompt_tokens=10))

    meter._run("text-embedding-3-small", 10, 0, flaky, lambda u: Usage(input_tokens=u.prompt_tokens), "embed")
    assert len(attempts) == 2 and meter.sink.calls == 1


class _SlowIngestArm(FullContextArm):
    """A full-context arm with a slow, recorded ingest, to test concurrent histories."""

    concurrent_prepare = True

    def __init__(self, delay: float = 0.2):
        super().__init__(100_000, 0.05)
        import threading

        self.delay = delay
        self.events: list[tuple[str, str]] = []
        self.active = self.peak = 0
        self._guard = threading.Lock()
        self._cache_lock = threading.Lock()

    def prepare(self, history):
        import time as _time

        with self._guard:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.events.append(("start", history.history_id))
        _time.sleep(self.delay)
        with self._guard:
            self.active -= 1
            self.events.append(("end", history.history_id))
        with self._cache_lock:
            super().prepare(history)
        return IngestStats(seconds=self.delay, cost=CostSink(usd=0.01, calls=1), stored_units=3)

    def context(self, item, history):
        with self._cache_lock:
            return super().context(item, history)


def _concurrent_run(prices, budget, cfg, tmp_path, arm, histories, ingest_workers):
    _, reader, judge = build(prices, budget, cfg)
    store = ResultStore(tmp_path)
    items = [make_item(h.history_id, n) for h in histories for n in range(2)]
    summary = run_chat(
        arm=arm, items=items, histories={h.history_id: h for h in histories},
        reader=reader, judge=judge, store=store, stage="pilot", run_id="r",
        ingest_workers=ingest_workers,
    )
    return summary, store


def test_histories_ingest_concurrently_and_every_answer_and_ingest_is_kept(prices, budget, cfg, tmp_path):
    import time as _time

    arm = _SlowIngestArm()
    histories = [make_history(f"h{i}") for i in range(4)]
    start = _time.perf_counter()
    summary, store = _concurrent_run(prices, budget, cfg, tmp_path, arm, histories, ingest_workers=4)
    elapsed = _time.perf_counter() - start
    assert arm.peak > 1 and elapsed < 4 * arm.delay
    assert summary.completed == 8 and summary.errors == 0
    assert sorted(r["item_id"] for r in store.read_items("A", "locomo")) == sorted(
        f"h{i}-{n:04d}" for i in range(4) for n in range(2)
    )
    ingests = store.read_ingest("A", "locomo")
    assert sorted(i["history_id"] for i in ingests) == ["h0", "h1", "h2", "h3"]
    assert all(i["cost"]["usd"] == 0.01 and i["stored_units"] == 3 for i in ingests)


def test_checkpoints_of_one_store_stay_in_order_while_other_stores_overlap(prices, budget, cfg, tmp_path):
    arm = _SlowIngestArm(delay=0.1)
    cuts = [make_history("c_s01"), make_history("c_s02"), make_history("c_s03")]
    for cut in cuts:
        cut.store_id = "c"
    other = make_history("other")
    _concurrent_run(prices, budget, cfg, tmp_path, arm, [*cuts, other], ingest_workers=4)
    order = [h for kind, h in arm.events if kind == "start" and h.startswith("c_s")]
    assert order == ["c_s01", "c_s02", "c_s03"]
    for a, b in zip(order, order[1:], strict=False):  # each checkpoint finishes before the next one starts
        assert arm.events.index(("end", a)) < arm.events.index(("start", b))
    assert arm.peak == 2  # the conversation's chain and the other history, never two checkpoints


def test_an_arm_without_concurrent_prepare_ingests_one_history_at_a_time(prices, budget, cfg, tmp_path):
    arm = _SlowIngestArm(delay=0.05)
    arm.concurrent_prepare = False
    histories = [make_history(f"h{i}") for i in range(3)]
    summary, _ = _concurrent_run(prices, budget, cfg, tmp_path, arm, histories, ingest_workers=4)
    assert arm.peak == 1 and summary.completed == 6


def test_a_budget_stop_in_one_history_stops_the_concurrent_run(prices, budget, cfg, tmp_path):
    class CappedArm(_SlowIngestArm):
        def prepare(self, history):
            if history.history_id == "h1":
                raise BudgetExceeded("cap")
            return super().prepare(history)

    histories = [make_history(f"h{i}") for i in range(3)]
    with pytest.raises(BudgetExceeded):
        _concurrent_run(prices, budget, cfg, tmp_path, CappedArm(delay=0.05), histories, ingest_workers=3)


def test_a_history_whose_ingestion_fails_is_recorded_and_the_run_continues(
    prices, budget, cfg, tmp_path
):
    from memstudy.arms.base import ArmError

    class StuckArm(FullContextArm):
        def prepare(self, history):
            if history.history_id == "h1":
                raise ArmError("ingestion failed for h1")
            return super().prepare(history)

    _, reader, judge = build(prices, budget, cfg)
    store = ResultStore(tmp_path)
    summary = run_chat(
        arm=StuckArm(100_000, 0.05), items=[make_item("h1", 0), make_item("h2", 0)],
        histories={"h1": make_history("h1"), "h2": make_history("h2")},
        reader=reader, judge=judge, store=store, stage="pilot", run_id="r",
    )
    assert summary.errors == 1 and summary.completed == 1
    assert store.read_errors("A", "locomo")[0]["error"] == "ingest_failed: ArmError"
