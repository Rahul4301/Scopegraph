"""Arm D: Supermemory, self-hosted server binary, one container per history.

Reported as "Supermemory (self-hosted, shared extraction model)", never as the hosted product:
the hosted platform runs proprietary extraction models, while the self-hosted server runs on the
model it is pointed at. The server is configured by its environment, not by this code, so the
extraction model and embedder match arm B:

    OPENAI_API_KEY=...  OPENAI_MODEL=<arms.B.extraction_model>
    SUPERMEMORY_EMBEDDING_PROVIDER=openai
    SUPERMEMORY_EMBEDDING_MODEL=<arms.B.embedding_model>  SUPERMEMORY_EMBEDDING_DIMENSIONS=1536
    supermemory-server            # release server-v0.0.8; see arms.D in study.yaml. The public repo
                                  # (MIT) holds SDKs, docs and plugins but not the server source.

Scoping: every write and every search carries container_tag = the history's store key
("<bench>_<history_id>") and nothing else. Ingestion is asynchronous on the server (an add
returns "queued"), so the arm waits until every document reports done before any query.

Cost: the server's own model calls are made with its own key, outside this process, so they are
not metered. Spend is estimated from our token counts at the shared models' verified prices and
charged to the ledger marked estimated. It counts input tokens only (extraction output and
summaries are unknown), so it is a lower bound.

Stored text (gold-in-store): everything the server holds for the container, from
POST /v4/memories/list (paged, latest and not-forgotten versions only). The SDK has no method for
it, so the arm calls the endpoint through the SDK's generic client.post. It is cached per
container until the next write. Like Mem0's, it is a verbatim test over paraphrased facts.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from memstudy.arms.base import ArmContext, IngestStats, fill_to_budget
from memstudy.budget import Budget, ModelPrice
from memstudy.metering import CostSink
from memstudy.schema import History, Item, Usage, render_session
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
    name = "D"

    def __init__(
        self,
        cfg: dict[str, Any],
        retrieval: dict[str, Any],
        client: Any,
        extraction_price: ModelPrice,
        embedding_price: ModelPrice,
        budget: Budget,
        stage: str,
        tag: dict[str, Any],
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        list_memories: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.retrieval = retrieval
        self.client = client
        self.extraction_price = extraction_price
        self.embedding_price = embedding_price
        self.budget = budget
        self.stage = stage
        self.tag = tag
        self._sleep = sleep
        self._clock = clock
        self._list_memories = list_memories
        self._ingested: set[str] = set()
        self._sessions_done: dict[str, set[str]] = {}
        self._stored_text: dict[str, str] = {}

    @staticmethod
    def _container(history: History) -> str:
        tag = history.store_key
        if not TAG_PATTERN.match(tag):
            raise ValueError(f"container tag {tag!r} is not allowed by Supermemory")
        return tag

    def _wait_until_done(self, doc_ids: list[str]) -> None:
        """Poll until every document, and any follow-up processing it reports, is done."""
        deadline = self._clock() + float(self.cfg["poll_timeout_seconds"])
        pending = list(doc_ids)
        while pending:
            still: list[str] = []
            for doc_id in pending:
                doc = self.client.documents.get(doc_id)
                dreaming = getattr(doc, "dreaming_status", None)
                if doc.status == "failed" or dreaming == "failed":
                    raise IngestFailed(f"document {doc_id} failed processing")
                if doc.status != "done" or dreaming not in (None, "done"):
                    still.append(doc_id)
            pending = still
            if pending:
                if self._clock() > deadline:
                    raise IngestFailed(f"{len(pending)} documents not done before the timeout")
                self._sleep(float(self.cfg["poll_seconds"]))

    def stored_memories(self, container: str) -> list[str]:
        """Latest, not-forgotten memories held for a container (all pages)."""
        if self._list_memories is not None:
            return self._list_memories(container)
        out: list[str] = []
        page = 1
        while True:
            resp = self.client.post(
                "/v4/memories/list",
                body={"containerTags": [container], "limit": 1000, "page": page},
                cast_to=object,
            )
            out += [
                str(e["memory"])
                for e in resp["memoryEntries"]
                if e.get("isLatest", True) and not e.get("isForgotten", False)
            ]
            if page >= int(resp["pagination"]["totalPages"]):
                return out
            page += 1

    def stored_text(self, container: str) -> str:
        if container not in self._stored_text:
            self._stored_text[container] = "\n".join(self.stored_memories(container))
        return self._stored_text[container]

    def _ingest_usd(self, tokens: int) -> float:
        """Estimated server-side spend: the history's tokens read by the extraction model and
        embedded once. Input only, so a lower bound."""
        return tokens * (self.extraction_price.input + self.embedding_price.input) / 1e6

    def prepare(self, history: History) -> IngestStats | None:
        if history.history_id in self._ingested:
            return None
        container = self._container(history)
        done = self._sessions_done.setdefault(container, set())
        new = [s for s in history.sessions if s.session_id not in done]
        tokens = sum(count_tokens(render_session(s)) for s in new)
        usd = self._ingest_usd(tokens)
        self.budget.guard(self.stage, usd)
        start = self._clock()
        ids: list[str] = []
        for session in new:
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
        done.update(s.session_id for s in new)
        self._stored_text.pop(container, None)
        self.budget.charge(
            self.stage,
            usd,
            model="supermemory-self-hosted",
            purpose="supermemory-ingest",
            estimated=True,
            memory_tokens=tokens,
            **self.tag,
        )
        self._ingested.add(history.history_id)
        seconds = self._clock() - start
        sink = CostSink(usd=usd, calls=len(ids), seconds=seconds, usage=Usage(input_tokens=tokens))
        return IngestStats(seconds=seconds, cost=sink, stored_units=len(done))

    def _search(self, query: str, container: str) -> list[Any]:
        found = self.client.search.memories(
            q=query,
            container_tag=container,
            search_mode=self.cfg["search_mode"],
            limit=int(self.retrieval["candidate_pool"]),
            threshold=float(self.cfg["threshold"]),
            rerank=bool(self.cfg.get("rerank", False)),
            rewrite_query=bool(self.cfg.get("rewrite_query", False)),
        )
        return [r for r in found.results if (r.memory or r.chunk)]

    def context(self, item: Item, history: History) -> ArmContext:
        if history.history_id not in self._ingested:
            raise RuntimeError(f"history {history.history_id} was not ingested before querying")
        container = self._container(history)
        usd = count_tokens(item.question) * self.embedding_price.input / 1e6
        self.budget.guard(self.stage, usd)
        start = self._clock()
        results = self._search(item.question, container)
        seconds = self._clock() - start
        self.budget.charge(
            self.stage,
            usd,
            model="supermemory-self-hosted",
            purpose="supermemory-query",
            estimated=True,
            **self.tag,
        )
        texts = [r.memory or r.chunk or "" for r in results]
        kept = fill_to_budget(texts, int(self.retrieval["token_budget"]))
        text = "\n".join(f"- {t}" for t in kept)
        return ArmContext(
            text=text,
            context_tokens=count_tokens(text),
            cache_prefix=False,
            retrieved=[str(getattr(r, "id", i)) for i, r in enumerate(results[: len(kept)])],
            candidates=len(results),
            retrieval_seconds=seconds,
            retrieval_cost=CostSink(usd=usd, calls=1, seconds=seconds),
            stored_text=self.stored_text(container),
        )
