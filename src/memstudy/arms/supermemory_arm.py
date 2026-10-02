"""Arm C: Supermemory (hosted API), one container per history.

Scoping: every write and every search carries `container_tag = user_id` ("<bench>_<history_id>")
and nothing else; the service isolates containers strictly. The singular container_tag is used,
never the deprecated plural, and a tag outside Supermemory's allowed pattern raises instead of
being rewritten.

Cost: the service's internal models are not visible to us, so spend is computed from its
published rate card and our own token counts (an estimate, labelled as such in prices.yaml) and
charged to the same ledger as everything else.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from memstudy.arms.base import ArmContext, IngestStats, fill_to_budget
from memstudy.budget import Budget, SupermemoryPrice
from memstudy.metering import CostSink
from memstudy.schema import History, Item, Session, Usage, render_session
from memstudy.tokens import count_tokens

TAG_PATTERN = re.compile(r"^[a-zA-Z0-9_:-]{1,100}$")
DATE_FORMATS = ("%I:%M %p on %d %B, %Y", "%Y/%m/%d (%a) %H:%M")


class IngestFailed(RuntimeError):
    pass


def parse_session_date(timestamp: str | None) -> str | None:
    """ISO date for a session timestamp, or None when the format is not recognised.
    LoCoMo ("1:56 pm on 8 May, 2023") and LongMemEval ("2023/05/20 (Sat) 10:00") are handled."""
    if not timestamp:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(timestamp.strip(), fmt).isoformat()
        except ValueError:
            continue
    return None


class SupermemoryArm:
    name = "C"

    def __init__(
        self,
        cfg: dict[str, Any],
        retrieval: dict[str, Any],
        client: Any,
        price: SupermemoryPrice,
        budget: Budget,
        stage: str,
        tag: dict[str, Any],
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.retrieval = retrieval
        self.client = client
        self.price = price
        self.budget = budget
        self.stage = stage
        self.tag = tag
        self._sleep = sleep
        self._clock = clock
        self._ingested: set[str] = set()

    @staticmethod
    def _container(history: History) -> str:
        tag = history.user_id
        if not TAG_PATTERN.match(tag):
            raise ValueError(f"container tag {tag!r} is not allowed by Supermemory")
        return tag

    def _wait_until_done(self, doc_ids: list[str]) -> None:
        """Poll until every document and its follow-up processing report done."""
        deadline = self._clock() + float(self.cfg["poll_timeout_seconds"])
        pending = list(doc_ids)
        while pending:
            still: list[str] = []
            for doc_id in pending:
                doc = self.client.documents.get(doc_id)
                if doc.status == "failed" or getattr(doc, "dreaming_status", None) == "failed":
                    raise IngestFailed(f"document {doc_id} failed processing")
                dreaming = getattr(doc, "dreaming_status", None)
                if doc.status != "done" or dreaming not in (None, "done"):
                    still.append(doc_id)
            pending = still
            if pending:
                if self._clock() > deadline:
                    raise IngestFailed(f"{len(pending)} documents not done before the timeout")
                self._sleep(float(self.cfg["poll_seconds"]))

    def _ingest_cost(self, sessions: list[Session]) -> tuple[int, float]:
        tokens = sum(count_tokens(render_session(s)) for s in sessions)
        return tokens, self.price.cost(tokens, 0, len(sessions))

    def prepare(self, history: History) -> IngestStats | None:
        if history.history_id in self._ingested:
            return None
        container = self._container(history)
        tokens, usd = self._ingest_cost(history.sessions)
        self.budget.guard(self.stage, usd)
        start = self._clock()
        ids: list[str] = []
        for session in history.sessions:
            kwargs: dict[str, Any] = {
                "content": render_session(session),
                "container_tag": container,
                "custom_id": f"{container}-s{session.session_id}",
                "task_type": self.cfg["task_type"],
                "metadata": {"session_id": session.session_id},
            }
            date = parse_session_date(session.timestamp)
            if date:
                kwargs["document_date"] = date
            ids.append(self.client.documents.add(**kwargs).id)
        self._wait_until_done(ids)
        self.budget.charge(
            self.stage,
            usd,
            model="supermemory",
            purpose="supermemory-ingest",
            estimated=True,
            memory_tokens=tokens,
            **self.tag,
        )
        self._ingested.add(history.history_id)
        sink = CostSink(
            usd=usd,
            calls=len(ids),
            seconds=self._clock() - start,
            usage=Usage(input_tokens=tokens),
        )
        return IngestStats(seconds=sink.seconds, cost=sink, stored_units=len(ids))

    def context(self, item: Item, history: History) -> ArmContext:
        if history.history_id not in self._ingested:
            raise RuntimeError(f"history {history.history_id} was not ingested before querying")
        usd = self.price.cost(0, 1, 0)
        self.budget.guard(self.stage, usd)
        start = self._clock()
        found = self.client.search.memories(
            q=item.question,
            container_tag=self._container(history),
            search_mode=self.cfg["search_mode"],
            limit=int(self.retrieval["candidate_pool"]),
        )
        seconds = self._clock() - start
        self.budget.charge(
            self.stage,
            usd,
            model="supermemory",
            purpose="supermemory-query",
            estimated=True,
            **self.tag,
        )
        results = [r for r in found.results if (r.memory or r.chunk)]
        kept = fill_to_budget([r.memory or r.chunk or "" for r in results], int(self.retrieval["token_budget"]))
        text = "\n".join(f"- {t}" for t in kept)
        return ArmContext(
            text=text,
            context_tokens=count_tokens(text),
            cache_prefix=False,
            retrieved=[str(getattr(r, "id", i)) for i, r in enumerate(results[: len(kept)])],
            candidates=len(results),
            retrieval_seconds=seconds,
            retrieval_cost=CostSink(usd=usd, calls=1, seconds=seconds),
        )
