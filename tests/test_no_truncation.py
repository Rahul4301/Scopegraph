import pytest
from conftest import FakeClient, make_history, make_item, make_response

from memstudy.arms.full_context import FullContextArm
from memstudy.arms.rag import chunk_history, chunk_session
from memstudy.llm import ModelCaller
from memstudy.phase0 import census
from memstudy.reader import Reader
from memstudy.runner import run_chat
from memstudy.schema import Session, Turn, render_transcript
from memstudy.store import ResultStore
from memstudy.tokens import DoesNotFit, count_tokens, require_fit


def test_full_context_sends_the_whole_transcript_verbatim():
    history = make_history()
    arm = FullContextArm(window=100_000, margin=0.05, queries_per_history={"h1": 5})
    ctx = arm.context(make_item(), history)
    assert ctx.text == render_transcript(history)
    assert ctx.cache_prefix is True and ctx.context_tokens == count_tokens(ctx.text)


def test_cache_breakpoint_only_when_the_history_is_queried_more_than_once():
    history = make_history()
    once = FullContextArm(100_000, 0.05, {"h1": 1}).context(make_item(), history)
    many = FullContextArm(100_000, 0.05, {"h1": 2}).context(make_item(), history)
    assert once.cache_prefix is False and many.cache_prefix is True


def test_history_over_the_window_raises_instead_of_truncating():
    history = make_history()
    arm = FullContextArm(window=20, margin=0.05)
    with pytest.raises(DoesNotFit) as err:
        arm.prepare(history)
    assert err.value.tokens > err.value.limit


def test_margin_shrinks_the_usable_window():
    require_fit("x", 95, 100, 0.0)
    with pytest.raises(DoesNotFit):
        require_fit("x", 96, 100, 0.05)


def test_runner_records_does_not_fit_and_never_calls_the_reader(prices, budget, cfg, tmp_path):
    client = FakeClient(lambda kw: make_response())
    reader = Reader(ModelCaller(client, prices["gpt-6-luna"], budget), cfg)
    store = ResultStore(tmp_path / "res")
    history = make_history("big")
    summary = run_chat(
        arm=FullContextArm(window=20, margin=0.05),
        items=[make_item("big", 0), make_item("big", 1)],
        histories={"big": history},
        reader=reader,
        judge=None,
        store=store,
        stage="pilot",
        run_id="r1",
    )
    assert client.responses.calls == []
    assert summary.does_not_fit == ["big"] and summary.errors == 2 and summary.completed == 0
    errors = store.read_errors("A", "locomo")
    assert {e["error"] for e in errors} == {"does_not_fit"}


def test_chunking_keeps_every_turn():
    history = make_history()
    chunks = chunk_history(history, chunk_tokens=30)
    joined = "\n".join(chunks)
    for session in history.sessions:
        for turn in session.turns:
            assert f"{turn.speaker}: {turn.text}" in joined


def test_oversized_turn_is_split_not_cut():
    long_text = " ".join(f"word{i}" for i in range(200))
    session = Session(session_id="1", turns=[Turn(speaker="Ann", text=long_text)])
    chunks = chunk_session(session, chunk_tokens=40)
    assert len(chunks) > 1
    body = "".join(c.split("\n", 1)[1] for c in chunks)
    assert body == f"Ann: {long_text}"


def test_census_flags_items_that_do_not_fit_or_cross_the_threshold(prices):
    luna = prices["gpt-6-luna"]
    small = make_history("small")
    result = census({"small": small}, [make_item("small")], luna, 0.05, overhead_tokens=0)
    assert result["n_exceed_window"] == 0 and result["n_over_long_context_threshold"] == 0
    tiny_window = luna.__class__(**{**luna.__dict__, "context_window": 10, "long_context_threshold": 5})
    flagged = census({"small": small}, [make_item("small")], tiny_window, 0.05, overhead_tokens=0)
    assert flagged["n_exceed_window"] == 1 and flagged["n_over_long_context_threshold"] == 1
