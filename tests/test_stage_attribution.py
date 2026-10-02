"""Gold-in-store logging and the extraction / retrieval / reading attribution."""

from types import SimpleNamespace as NS

import pytest
from conftest import FakeClient, make_history, make_item, make_response

from memstudy.arms.base import ArmContext
from memstudy.arms.mem0_arm import Mem0Arm
from memstudy.budget import Budget
from memstudy.judge import Judge
from memstudy.llm import ModelCaller
from memstudy.metering import CostSink, Meter
from memstudy.reader import Reader
from memstudy.runner import run_chat, stage_attribution
from memstudy.scoring import gold_in_store
from memstudy.store import MemoryStore


def test_gold_in_store_is_a_normalized_substring_test():
    assert gold_in_store("Ann adopted a dog named Alpha in May.", ["alpha"])
    assert not gold_in_store("Ann likes cats.", ["alpha"])
    assert gold_in_store("went to the beach on 8 May, 2023", ["8 May 2023"])


class StoreArm:
    """Arm with a store: context carries what it holds for the history."""

    name = "B"

    def __init__(self, stored: str | None) -> None:
        self.stored = stored

    def prepare(self, history):
        return None

    def context(self, item, history):
        return ArmContext(text="- alpha", context_tokens=3, cache_prefix=False, stored_text=self.stored)


def build(prices, budget, cfg):
    def handler(kw):
        if kw["model"] == "gpt-6-luna":
            return make_response(text="alpha")
        return make_response(text='{"verdict": "CORRECT"}', model="nano")

    client = FakeClient(handler)
    return (
        Reader(ModelCaller(client, prices["gpt-6-luna"], budget), cfg),
        Judge(ModelCaller(client, prices["gpt-5-nano"], budget), cfg),
    )


def run(arm, prices, budget, cfg, items):
    reader, judge = build(prices, budget, cfg)
    store = MemoryStore()
    run_chat(
        arm=arm, items=items, histories={"h1": make_history("h1")}, reader=reader, judge=judge,
        store=store, stage="pilot", run_id="r",
    )
    return store.read_items("B", "locomo")


def test_record_carries_gold_in_store_for_an_arm_with_a_store(prices, budget, cfg):
    stored = run(StoreArm("Ann has a dog named Alpha"), prices, budget, cfg, [make_item("h1")])[0]
    missing = run(StoreArm("Ann has a cat"), prices, budget, cfg, [make_item("h1", 1)])[0]
    assert stored["gold_in_store"] is True and missing["gold_in_store"] is False


def test_arms_without_a_store_and_abstention_items_have_none(prices, budget, cfg):
    assert run(StoreArm(None), prices, budget, cfg, [make_item("h1")])[0]["gold_in_store"] is None
    item = make_item("h1", 2).model_copy(update={"meta": {"abstention": True}})
    assert run(StoreArm("alpha"), prices, budget, cfg, [item])[0]["gold_in_store"] is None


def row(in_store, in_context, correct):
    return {
        "gold_in_store": in_store,
        "metrics": {"gold_in_context": in_context},
        "correct": correct,
    }


def test_stage_attribution_splits_failures_by_stage():
    records = [
        row(False, False, False),  # extraction
        row(True, False, False),   # retrieval
        row(True, True, False),    # reading
        row(True, True, True),
        row(True, True, True),
    ]
    got = stage_attribution(records)
    assert got["n"] == 5 and got["not_stored"] == 1 and got["stored_not_retrieved"] == 1
    assert got["stored_retrieved_graded"] == 3 and got["stored_retrieved_correct"] == 2
    assert got["correct_overall"] == 2


def test_stage_attribution_is_none_without_a_store():
    assert stage_attribution([{"gold_in_store": None, "metrics": {"gold_in_context": True}}]) is None
    assert stage_attribution([]) is None


class FakeMemory:
    """Mem0 stand-in: records add calls and stores each message as one memory."""

    def __init__(self):
        self.adds: list[tuple[int, str]] = []
        self.rows: dict[str, list[str]] = {}
        self.llm = NS(client=None)
        self.embedding_model = NS(client=None)

    def add(self, messages, user_id, metadata=None, infer=True):
        self.adds.append((len(messages), user_id))
        self.rows.setdefault(user_id, []).extend(m["content"] for m in messages)

    def get_all(self, filters, top_k):
        return {"results": [{"memory": t} for t in self.rows.get(filters["user_id"], [])]}

    def search(self, query, filters, top_k, threshold, rerank):
        return {"results": [{"id": str(i), "memory": t} for i, t in enumerate(self.rows.get(filters["user_id"], []))]}


def mem0(cfg, prices, tmp_path, **b):
    meter = Meter(Budget(tmp_path / "l.jsonl"), prices, "pilot", CostSink(), {})
    memory = FakeMemory()
    return Mem0Arm({**cfg["arms"]["B"], **b}, cfg["retrieval"], meter, memory), memory


def test_mem0_context_carries_everything_stored_for_the_history(cfg, prices, tmp_path):
    arm, _ = mem0(cfg, prices, tmp_path)
    h = make_history("conv-1")
    arm.prepare(h)
    ctx = arm.context(make_item("conv-1"), h)
    assert "dog named alpha" in ctx.stored_text and "swim" in ctx.stored_text


@pytest.mark.parametrize(("granularity", "expected"), [("turn", [1, 1, 1]), ("ten", [2, 1]), ("session", [2, 1])])
def test_write_granularity_sets_messages_per_add(cfg, prices, tmp_path, granularity, expected):
    arm, memory = mem0(cfg, prices, tmp_path, write_granularity=granularity)
    arm.prepare(make_history("conv-1"))
    assert [n for n, _ in memory.adds] == expected


def test_granularity_splits_long_sessions_only_for_turn_and_ten(cfg, prices, tmp_path):
    from memstudy.schema import History, Session, Turn

    long = History(
        history_id="c", bench="locomo",
        sessions=[Session(session_id="1", turns=[Turn(speaker="A", text=f"t{i}") for i in range(25)])],
    )
    for granularity, expected in (("turn", 25), ("ten", 3), ("session", 1)):
        arm, memory = mem0(cfg, prices, tmp_path, write_granularity=granularity)
        arm.prepare(long)
        assert len(memory.adds) == expected


def test_checkpoints_share_one_store_and_add_only_new_sessions(cfg, prices, tmp_path):
    from memstudy.schema import History

    arm, memory = mem0(cfg, prices, tmp_path)
    full = make_history("conv-1")
    early = History(history_id="conv-1@s01", bench="locomo", sessions=full.sessions[:1], store_id="conv-1")
    arm.prepare(early)
    assert [n for n, _ in memory.adds] == [2]
    ctx = arm.context(make_item("conv-1"), early)
    assert "swim" not in ctx.stored_text  # session 2 is still in the future
    arm.prepare(full)
    assert [n for n, _ in memory.adds] == [2, 1]  # only session 2 was written
    assert {u for _, u in memory.adds} == {"locomo_conv-1"}
    assert "swim" in arm.context(make_item("conv-1"), full).stored_text
