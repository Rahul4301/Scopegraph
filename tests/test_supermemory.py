"""Supermemory arm (self-hosted) with a fake server that enforces container isolation."""

from types import SimpleNamespace as NS

import pytest
from conftest import make_history, make_item

from memstudy.arms.supermemory_arm import IngestFailed, SupermemoryArm, parse_session_date
from memstudy.budget import Budget, BudgetExceeded
from memstudy.schema import History


class FakeSupermemory:
    """Stores documents per container_tag; search sees only the queried container."""

    def __init__(self, statuses=("done",)):
        self.docs: dict[str, list[dict]] = {}
        self.calls: list[dict] = []
        self.searches: list[dict] = []
        self.statuses = list(statuses)
        self.documents = NS(add=self._add, get=self._get)
        self.search = NS(memories=self._search)
        self.list_calls: list[dict] = []
        self.forgotten: set[str] = set()

    def _add(self, **kw):
        self.calls.append(kw)
        self.docs.setdefault(kw["container_tag"], []).append(kw)
        return NS(id=f"doc-{len(self.calls)}", status="queued")

    def _get(self, doc_id):
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return NS(id=doc_id, status=status, dreaming_status="done" if status == "done" else None)

    def post(self, path, body, cast_to):
        """POST /v4/memories/list: two entries per page, one extra forgotten and one old version."""
        assert path == "/v4/memories/list" and len(body["containerTags"]) == 1
        self.list_calls.append(body)
        entries = [
            {"memory": d["content"], "isLatest": True, "isForgotten": d["custom_id"] in self.forgotten}
            for d in self.docs.get(body["containerTags"][0], [])
        ] + [{"memory": "old version", "isLatest": False, "isForgotten": False}]
        size = 2
        chunk = entries[(body["page"] - 1) * size : body["page"] * size]
        return {
            "memoryEntries": chunk,
            "pagination": {
                "currentPage": body["page"],
                "totalPages": -(-len(entries) // size),
                "totalItems": len(entries),
            },
        }

    def _search(self, q, container_tag, search_mode, limit, threshold, rerank, rewrite_query):
        self.searches.append(
            {"mode": search_mode, "threshold": threshold, "rerank": rerank, "rewrite": rewrite_query, "limit": limit}
        )
        hits = [
            NS(memory=d["content"], chunk=None, id=d["custom_id"], similarity=1.0)
            for d in self.docs.get(container_tag, [])
        ]
        return NS(results=hits[:limit])


def make_arm(cfg, prices, tmp_path, client, **kw):
    d = cfg["arms"]["D"]
    return SupermemoryArm(
        d, cfg["retrieval"], client, prices[d["extraction_model"]], prices[d["embedding_model"]],
        Budget(tmp_path / "l.jsonl"), "pilot", {"arm": "D"}, sleep=lambda s: None, **kw,
    )


def test_search_under_one_container_returns_nothing_from_another(cfg, prices, tmp_path):
    client = FakeSupermemory()
    arm = make_arm(cfg, prices, tmp_path, client)
    h1, h2 = make_history("conv-1", "locomo", "alpha"), make_history("conv-2", "locomo", "bravo")
    arm.prepare(h1)
    arm.prepare(h2)
    own = arm.context(make_item("conv-1"), h1)
    assert "alpha" in own.text and "bravo" not in own.text
    assert "alpha" not in arm.context(make_item("conv-2"), h2).text
    assert {c["container_tag"] for c in client.calls} == {"locomo_conv-1", "locomo_conv-2"}
    assert all("container_tags" not in c for c in client.calls)


def test_every_write_is_one_session_with_the_memory_task_type_and_date(cfg, prices, tmp_path):
    client = FakeSupermemory()
    make_arm(cfg, prices, tmp_path, client).prepare(make_history("conv-1", "locomo"))
    first = client.calls[0]
    assert first["task_type"] == "memory" and first["custom_id"] == "locomo_conv-1-s1"
    assert first["document_date"] == "2023-05-01T13:00:00"
    assert "[Session 1 | 1:00 pm on 1 May, 2023]" in first["content"]
    assert len(client.calls) == 2


def test_search_uses_the_registered_mode_with_rerank_and_rewrite_off(cfg, prices, tmp_path):
    client = FakeSupermemory()
    arm = make_arm(cfg, prices, tmp_path, client)
    h = make_history("conv-1")
    arm.prepare(h)
    arm.context(make_item("conv-1"), h)
    assert client.searches == [
        {
            "mode": "memories",
            "threshold": 0.6,
            "rerank": False,
            "rewrite": False,
            "limit": cfg["retrieval"]["candidate_pool"],
        }
    ]


def test_container_tag_outside_the_allowed_pattern_raises(cfg, prices, tmp_path):
    arm = make_arm(cfg, prices, tmp_path, FakeSupermemory())
    with pytest.raises(ValueError, match="not allowed"):
        arm.prepare(make_history("a.b", "locomo"))


def test_ingestion_waits_until_every_document_is_done(cfg, prices, tmp_path):
    client = FakeSupermemory(statuses=("queued", "embedding", "done"))
    make_arm(cfg, prices, tmp_path, client).prepare(make_history("conv-1", "locomo"))
    assert client.statuses == ["done"]


def test_failed_or_stuck_ingestion_raises(cfg, prices, tmp_path):
    with pytest.raises(IngestFailed, match="failed"):
        make_arm(cfg, prices, tmp_path, FakeSupermemory(statuses=("failed",))).prepare(make_history("conv-1"))
    ticks = iter(range(0, 10_000, 600))
    stuck = make_arm(cfg, prices, tmp_path, FakeSupermemory(statuses=("queued",)), clock=lambda: next(ticks))
    with pytest.raises(IngestFailed, match="timeout"):
        stuck.prepare(make_history("conv-2"))


def test_cost_is_an_estimate_from_the_shared_models_prices_and_marked_so(cfg, prices, tmp_path):
    arm = make_arm(cfg, prices, tmp_path, FakeSupermemory())
    history = make_history("conv-1")
    stats = arm.prepare(history)
    tokens = stats.cost.usage.input_tokens
    d = cfg["arms"]["D"]
    expected = tokens * (prices[d["extraction_model"]].input + prices[d["embedding_model"]].input) / 1e6
    assert stats.cost.usd == pytest.approx(expected)
    ctx = arm.context(make_item("conv-1"), history)
    assert 0 < ctx.retrieval_cost.usd < 1e-6
    rows = arm.budget.ledger_path.read_text().splitlines()
    assert len(rows) == 2 and all('"estimated": true' in r for r in rows)


def test_budget_guard_blocks_ingestion_before_any_write(cfg, prices, tmp_path):
    client = FakeSupermemory()
    d = cfg["arms"]["D"]
    arm = SupermemoryArm(
        d, cfg["retrieval"], client, prices[d["extraction_model"]], prices[d["embedding_model"]],
        Budget(tmp_path / "l.jsonl", stage_caps={"pilot": 1e-12}), "pilot", {}, sleep=lambda s: None,
    )
    with pytest.raises(BudgetExceeded):
        arm.prepare(make_history("conv-1", "locomo"))
    assert client.calls == []


def test_context_refuses_a_history_that_was_not_ingested(cfg, prices, tmp_path):
    arm = make_arm(cfg, prices, tmp_path, FakeSupermemory())
    with pytest.raises(RuntimeError):
        arm.context(make_item("conv-1"), make_history("conv-1", "locomo"))


def test_checkpoints_share_one_container_and_write_only_new_sessions(cfg, prices, tmp_path):
    client = FakeSupermemory()
    arm = make_arm(cfg, prices, tmp_path, client)
    full = make_history("conv-1")
    early = History(history_id="conv-1@s01", bench="locomo", sessions=full.sessions[:1], store_id="conv-1")
    arm.prepare(early)
    arm.prepare(full)
    assert [c["custom_id"] for c in client.calls] == ["locomo_conv-1-s1", "locomo_conv-1-s2"]
    assert {c["container_tag"] for c in client.calls} == {"locomo_conv-1"}


def test_stored_text_pages_through_the_memory_list_and_keeps_only_live_latest_memories(cfg, prices, tmp_path):
    client = FakeSupermemory()
    arm = make_arm(cfg, prices, tmp_path, client)
    h = make_history("conv-1")
    arm.prepare(h)
    client.forgotten.add("locomo_conv-1-s2")
    text = arm.context(make_item("conv-1"), h).stored_text
    assert "alpha" in text and "swim" not in text and "old version" not in text
    assert [c["page"] for c in client.list_calls] == [1, 2]  # three entries, two per page


def test_stored_text_is_listed_once_per_container_until_the_next_write(cfg, prices, tmp_path):
    client = FakeSupermemory()
    arm = make_arm(cfg, prices, tmp_path, client)
    full = make_history("conv-1")
    early = History(history_id="conv-1@s01", bench="locomo", sessions=full.sessions[:1], store_id="conv-1")
    arm.prepare(early)
    arm.context(make_item("conv-1"), early)
    arm.context(make_item("conv-1"), early)
    assert len(client.list_calls) == 1  # one listing (one page), not one per question
    arm.prepare(full)
    arm.context(make_item("conv-1"), full)
    assert len(client.list_calls) == 3  # relisted (two pages) after the new session was written


def test_an_injected_lister_replaces_the_endpoint(cfg, prices, tmp_path):
    client = FakeSupermemory()
    h = make_history("conv-1")
    arm = make_arm(cfg, prices, tmp_path, client, list_memories=lambda tag: [f"fact in {tag}"])
    arm.prepare(h)
    assert arm.context(make_item("conv-1"), h).stored_text == "fact in locomo_conv-1"
    assert client.list_calls == []


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
