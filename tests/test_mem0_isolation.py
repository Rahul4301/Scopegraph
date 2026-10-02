"""Mem0 scoping: a query under one user_id returns zero memories from another.

Uses the real Mem0 Memory class and a real local Qdrant store. The embedder is replaced by one
that gives every text the same vector, the worst case for leakage: if scoping failed, any search
would match every memory regardless of user.
"""

import os

import pytest
from conftest import make_history, make_item

from memstudy.arms.mem0_arm import Mem0Arm, build_memory_config, session_messages
from memstudy.budget import Budget
from memstudy.metering import CostSink, Meter
from memstudy.schema import Session, Turn
from memstudy.tokens import count_tokens

DIMS = 1536


class ConstantEmbedder:
    def embed(self, text, memory_action=None):
        return [1.0] + [0.0] * (DIMS - 1)

    def embed_batch(self, texts, memory_action="add"):
        return [self.embed(t) for t in texts]


@pytest.fixture
def arm(tmp_path, cfg, prices, monkeypatch):
    monkeypatch.setenv("MEM0_TELEMETRY", "False")
    monkeypatch.setenv("MEM0_DIR", str(tmp_path / "mem0"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    from mem0 import Memory
    from mem0.configs.base import MemoryConfig

    config = build_memory_config(
        cfg["arms"]["B"], str(tmp_path / "qdrant"), str(tmp_path / "history.db")
    )
    memory = Memory(MemoryConfig(**config))
    memory.embedding_model = ConstantEmbedder()
    meter = Meter(Budget(tmp_path / "l.jsonl"), prices, "pilot", CostSink(), {})
    return Mem0Arm(cfg["arms"]["B"], cfg["retrieval"], meter, memory, infer=False)


def test_search_under_one_user_returns_nothing_from_another(arm):
    h1 = make_history("conv-1", "locomo", "alpha")
    h2 = make_history("conv-2", "locomo", "bravo")
    arm.prepare(h1)
    arm.prepare(h2)

    own = arm.search("dog", h1.user_id)
    other = arm.search("dog", h2.user_id)
    assert own and other
    assert all("alpha" in r["memory"] or "wonderful" in r["memory"] for r in own)
    assert not any("bravo" in r["memory"] for r in own)
    assert not any("alpha" in r["memory"] for r in other)


def test_unknown_user_id_returns_zero_memories(arm):
    arm.prepare(make_history("conv-1", "locomo", "alpha"))
    assert arm.search("dog", "locomo_never-ingested") == []


def test_same_history_id_in_another_bench_is_a_different_user(arm):
    a = make_history("shared", "locomo", "alpha")
    b = make_history("shared", "longmemeval", "bravo")
    arm.prepare(a)
    arm.prepare(b)
    assert a.user_id != b.user_id
    assert not any("bravo" in r["memory"] for r in arm.search("dog", a.user_id))


def test_context_refuses_a_history_that_was_not_ingested(arm):
    with pytest.raises(RuntimeError):
        arm.context(make_item("conv-1"), make_history("conv-1"))


def test_session_messages_carry_the_date_and_map_speakers():
    session = Session(
        session_id="1",
        timestamp="1 May 2023",
        turns=[Turn(speaker="Ann", text="hi"), Turn(speaker="Bob", text="yo")],
    )
    msgs = session_messages(session, {})
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "[1 May 2023] Ann: hi"


def test_longmemeval_roles_are_kept():
    session = Session(session_id="s", turns=[Turn(speaker="assistant", text="a")])
    assert session_messages(session, {}) == [{"role": "assistant", "content": "a"}]


def test_config_keeps_mem0_defaults_and_writes_nothing_under_home(cfg):
    b = build_memory_config(cfg["arms"]["B"], ".cache/v", ".cache/h.db")
    assert b["llm"] == {"provider": "openai", "config": {"model": "gpt-5-mini"}}
    assert b["embedder"]["config"]["model"] == "text-embedding-3-small"
    assert b["history_db_path"] == ".cache/h.db"
    assert os.path.expanduser("~/.mem0") not in b["vector_store"]["config"]["path"]


def test_mem0_context_is_filled_to_the_shared_token_budget(arm, cfg):
    history = make_history("conv-1", "locomo", "alpha")
    arm.prepare(history)
    top = arm.search("dog", history.user_id)[0]["memory"]
    arm.retrieval = {**cfg["retrieval"], "token_budget": count_tokens(top) + 3}
    ctx = arm.context(make_item("conv-1"), history)
    assert len(ctx.retrieved) == 1 and ctx.candidates > 1
    assert ctx.text == f"- {top}"
