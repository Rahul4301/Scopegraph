"""Supermemory arm with a fake service that enforces container isolation."""

from types import SimpleNamespace as NS

import pytest
from conftest import ROOT, make_history, make_item

from memstudy.arms.base import fill_to_budget
from memstudy.arms.supermemory_arm import IngestFailed, SupermemoryArm, parse_session_date
from memstudy.budget import Budget, BudgetExceeded, load_supermemory_price


class FakeSupermemory:
    """Stores documents per container_tag; search sees only the queried container."""

    def __init__(self, statuses=("done",)):
        self.docs: dict[str, list[dict]] = {}
        self.calls: list[dict] = []
        self.statuses = list(statuses)
        self.documents = NS(add=self._add, get=self._get)
        self.search = NS(memories=self._search)

    def _add(self, **kw):
        self.calls.append(kw)
        self.docs.setdefault(kw["container_tag"], []).append(kw)
        return NS(id=f"doc-{len(self.calls)}", status="queued")

    def _get(self, doc_id):
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return NS(id=doc_id, status=status, dreaming_status="done" if status == "done" else None)

    def _search(self, q, container_tag, search_mode, limit):
        assert "container_tags" not in self.calls
        hits = [NS(memory=None, chunk=d["content"], id=d["custom_id"], similarity=1.0)
                for d in self.docs.get(container_tag, [])]
        return NS(results=hits[:limit])


@pytest.fixture
def price():
    return load_supermemory_price(ROOT / "configs" / "prices.yaml")


def make_arm(cfg, price, tmp_path, client, **kw):
    return SupermemoryArm(
        cfg["arms"]["C"], cfg["retrieval"], client, price, Budget(tmp_path / "l.jsonl"),
        "pilot", {"arm": "C"}, sleep=lambda s: None, **kw,
    )


def test_search_under_one_container_returns_nothing_from_another(cfg, price, tmp_path):
    client = FakeSupermemory()
    arm = make_arm(cfg, price, tmp_path, client)
    h1, h2 = make_history("conv-1", "locomo", "alpha"), make_history("conv-2", "locomo", "bravo")
    arm.prepare(h1)
    arm.prepare(h2)
    own = arm.context(make_item("conv-1"), h1)
    assert "alpha" in own.text and "bravo" not in own.text
    assert "alpha" not in arm.context(make_item("conv-2"), h2).text
    assert {c["container_tag"] for c in client.calls} == {"locomo_conv-1", "locomo_conv-2"}
    assert all("container_tags" not in c for c in client.calls)


def test_every_write_is_one_session_with_the_memory_task_type_and_date(cfg, price, tmp_path):
    client = FakeSupermemory()
    make_arm(cfg, price, tmp_path, client).prepare(make_history("conv-1", "locomo"))
    first = client.calls[0]
    assert first["task_type"] == "memory" and first["custom_id"] == "locomo_conv-1-s1"
    assert first["document_date"] == "2023-05-01T13:00:00"
    assert "[Session 1 | 1:00 pm on 1 May, 2023]" in first["content"]
    assert len(client.calls) == 2


def test_container_tag_outside_the_allowed_pattern_raises(cfg, price, tmp_path):
    arm = make_arm(cfg, price, tmp_path, FakeSupermemory())
    with pytest.raises(ValueError, match="not allowed"):
        arm.prepare(make_history("a.b", "locomo"))


def test_ingestion_waits_until_every_document_is_done(cfg, price, tmp_path):
    client = FakeSupermemory(statuses=("queued", "embedding", "done"))
    make_arm(cfg, price, tmp_path, client).prepare(make_history("conv-1", "locomo"))
    assert client.statuses == ["done"]


def test_failed_or_stuck_ingestion_raises(cfg, price, tmp_path):
    with pytest.raises(IngestFailed, match="failed"):
        make_arm(cfg, price, tmp_path, FakeSupermemory(statuses=("failed",))).prepare(
            make_history("conv-1", "locomo")
        )
    ticks = iter(range(0, 10_000, 600))
    stuck = make_arm(cfg, price, tmp_path, FakeSupermemory(statuses=("queued",)), clock=lambda: next(ticks))
    with pytest.raises(IngestFailed, match="timeout"):
        stuck.prepare(make_history("conv-2", "locomo"))


def test_cost_is_charged_from_the_rate_card_and_marked_estimated(cfg, price, tmp_path):
    arm = make_arm(cfg, price, tmp_path, FakeSupermemory())
    history = make_history("conv-1", "locomo")
    stats = arm.prepare(history)
    ctx = arm.context(make_item("conv-1"), history)
    assert stats.cost.usd == pytest.approx(price.cost(stats.cost.usage.input_tokens, 0, 2))
    assert ctx.retrieval_cost.usd == pytest.approx(5e-6)
    rows = arm.budget.ledger_path.read_text().splitlines()
    assert len(rows) == 2 and all('"estimated": true' in r for r in rows)


def test_budget_guard_blocks_ingestion_before_any_write(cfg, price, tmp_path):
    client = FakeSupermemory()
    arm = SupermemoryArm(
        cfg["arms"]["C"], cfg["retrieval"], client, price,
        Budget(tmp_path / "l.jsonl", stage_caps={"pilot": 1e-9}), "pilot", {}, sleep=lambda s: None,
    )
    with pytest.raises(BudgetExceeded):
        arm.prepare(make_history("conv-1", "locomo"))
    assert client.calls == []


def test_context_refuses_a_history_that_was_not_ingested(cfg, price, tmp_path):
    arm = make_arm(cfg, price, tmp_path, FakeSupermemory())
    with pytest.raises(RuntimeError):
        arm.context(make_item("conv-1"), make_history("conv-1", "locomo"))


@pytest.mark.parametrize(
    ("stamp", "iso"),
    [
        ("1:56 pm on 8 May, 2023", "2023-05-08T13:56:00"),
        ("2023/05/20 (Sat) 10:00", "2023-05-20T10:00:00"),
        ("next tuesday", None),
        (None, None),
    ],
)
def test_session_dates_are_parsed_or_left_out(stamp, iso):
    assert parse_session_date(stamp) == iso


def test_fill_to_budget_keeps_whole_items_in_rank_order_and_never_cuts():
    items = ["one two three", "four five six", "seven eight nine"]
    assert fill_to_budget(items, 100) == items
    two = fill_to_budget(items, 2 * (3 + 2) + 1)
    assert two == items[:2]
    assert fill_to_budget(["word " * 50], 10) == []
