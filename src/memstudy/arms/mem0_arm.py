"""Arm B: Mem0 (open source, pinned version), one stable user_id per history.

Scoping: every add and every search carries user_id = "<bench>_<history_id>" and nothing else,
so a search under one user_id can only see memories written under that same user_id.
The OSS library does not support the Platform-only timestamp argument, so the session date is
written into each message instead.
"""

from __future__ import annotations

import os
import time
from typing import Any

from memstudy.arms.base import ArmContext, IngestStats, fill_to_budget
from memstudy.chunking import chunk_document
from memstudy.metering import CostSink, Meter
from memstudy.schema import History, Item, Session
from memstudy.tokens import count_tokens

NLP_MODEL = "en_core_web_sm"


def require_nlp() -> None:
    """Mem0 hybrid retrieval needs spaCy. Without it Mem0 silently degrades to semantic-only
    search, which would understate arm B, so the arm refuses to start instead."""
    import spacy

    if not spacy.util.is_package(NLP_MODEL):
        raise RuntimeError(
            f"spaCy model {NLP_MODEL} is not installed; run: python -m spacy download {NLP_MODEL}"
        )


def session_messages(session: Session) -> list[dict[str, str]]:
    """Messages for one session, each carrying the session date. LongMemEval keeps its real
    user/assistant roles. LoCoMo speakers are peers, so every turn is sent as a user message
    prefixed with the speaker's name: Mem0's default extraction prompt only reads user messages
    and would drop everything one of the two speakers said if that speaker were the assistant."""
    stamp = f"[{session.timestamp}] " if session.timestamp else ""
    out: list[dict[str, str]] = []
    for turn in session.turns:
        if turn.speaker in ("user", "assistant"):
            role, content = turn.speaker, turn.text
        else:
            role, content = "user", f"{turn.speaker}: {turn.text}"
        out.append({"role": role, "content": f"{stamp}{content}"})
    return out


def build_memory_config(cfg: dict[str, Any], vector_path: str, history_db: str) -> dict[str, Any]:
    llm: dict[str, Any] = {"model": cfg["extraction_model"]}
    if cfg.get("extraction_reasoning_effort"):
        # Mem0 only recognises its own list of reasoning models. Declaring the model as one makes
        # it send just messages, response_format and reasoning_effort.
        llm |= {"is_reasoning_model": True, "reasoning_effort": cfg["extraction_reasoning_effort"]}
    return {
        "history_db_path": history_db,
        "llm": {"provider": "openai", "config": llm},
        "embedder": {"provider": "openai", "config": {"model": cfg["embedding_model"]}},
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "memstudy",
                "path": vector_path,
                "embedding_model_dims": 1536,
            },
        },
    }


class Mem0Arm:
    name = "B"

    def __init__(
        self,
        cfg: dict[str, Any],
        retrieval: dict[str, Any],
        meter: Meter,
        memory: Any,
        infer: bool = True,
    ) -> None:
        self.cfg = cfg
        self.retrieval = retrieval
        self.meter = meter
        self.memory = memory
        self.infer = infer
        self.ingest_sink = CostSink()
        self.query_sink = CostSink()
        self._ingested: set[str] = set()
        # Sessions already written per store, so as-of checkpoints add only the new sessions.
        self._sessions_done: dict[str, set[str]] = {}
        self._documents_done: set[str] = set()
        self._stored_text: dict[str, str] = {}

    @classmethod
    def create(
        cls, cfg: dict[str, Any], retrieval: dict[str, Any], meter: Meter, store_dir: str
    ) -> Mem0Arm:
        # Mem0 reads these at import: no telemetry, and no files under the home directory.
        os.environ["MEM0_TELEMETRY"] = "False"
        os.environ["MEM0_DIR"] = store_dir
        from mem0 import Memory
        from mem0.configs.base import MemoryConfig

        require_nlp()
        config = build_memory_config(cfg, f"{store_dir}/qdrant", f"{store_dir}/history.db")
        memory = Memory(MemoryConfig(**config))
        meter.wrap(memory.llm.client)
        meter.wrap(memory.embedding_model.client)
        return cls(cfg, retrieval, meter, memory)

    def stored_memories(self, user_id: str) -> list[str]:
        found = self.memory.get_all(filters={"user_id": user_id}, top_k=100_000)
        return [str(r["memory"]) for r in found["results"]]

    def stored_count(self, user_id: str) -> int:
        return len(self.stored_memories(user_id))

    def stored_text(self, user_id: str) -> str:
        """All stored facts for one store, cached until the next write to that store."""
        if user_id not in self._stored_text:
            self._stored_text[user_id] = "\n".join(self.stored_memories(user_id))
        return self._stored_text[user_id]

    def _batch_size(self, n_messages: int) -> int:
        """Messages per add call: one turn, ten turns (the library-style default), or a whole
        session. The pre-registered default is ten."""
        granularity = self.cfg.get("write_granularity", "ten")
        if granularity == "turn":
            return 1
        if granularity == "session":
            return max(1, n_messages)
        return int(self.cfg["turns_per_add"])

    def prepare(self, history: History) -> IngestStats | None:
        if history.history_id in self._ingested:
            return None
        user_id = history.store_key
        before = self.ingest_sink.snapshot()
        start = time.perf_counter()
        done = self._sessions_done.setdefault(user_id, set())
        with self.meter.use_sink(self.ingest_sink):
            if history.document is not None and user_id not in self._documents_done:
                # A verbatim document (MemoryAgentBench): one fixed-size chunk per add call.
                size = int(self.cfg["document_chunk_tokens"])
                for n, chunk in enumerate(chunk_document(history.document, size)):
                    self.memory.add(
                        [{"role": "user", "content": chunk}],
                        user_id=user_id,
                        metadata={"chunk": n},
                        infer=self.infer,
                    )
                self._documents_done.add(user_id)
            batches = [
                (session, messages[i : i + self._batch_size(len(messages))])
                for session in history.sessions
                if session.session_id not in done
                for messages in [session_messages(session)]
                for i in range(0, len(messages), self._batch_size(len(messages)))
            ]
            for n, (session, batch) in enumerate(batches, start=1):
                self.memory.add(
                    batch,
                    user_id=user_id,
                    metadata={"session_id": session.session_id},
                    infer=self.infer,
                )
                if n % 5 == 0 or n == len(batches):
                    # Each add is a model call and takes seconds; say so rather than look hung.
                    print(
                        f"  ingesting {history.history_id}: {n}/{len(batches)} adds, "
                        f"{time.perf_counter() - start:.0f}s",
                        flush=True,
                    )
            done.update(s.session_id for s in history.sessions)
        self._stored_text.pop(user_id, None)
        self._ingested.add(history.history_id)
        return IngestStats(
            seconds=time.perf_counter() - start,
            cost=self.ingest_sink.since(before),
            stored_units=self.stored_count(user_id),
        )

    def search(self, query: str, user_id: str) -> list[dict[str, Any]]:
        found = self.memory.search(
            query,
            filters={"user_id": user_id},
            top_k=int(self.retrieval["candidate_pool"]),
            threshold=float(self.cfg["threshold"]),
            rerank=bool(self.cfg["rerank"]),
        )
        return list(found["results"])

    def context(self, item: Item, history: History) -> ArmContext:
        if history.history_id not in self._ingested:
            raise RuntimeError(f"history {history.history_id} was not ingested before querying")
        before = self.query_sink.snapshot()
        start = time.perf_counter()
        with self.meter.use_sink(self.query_sink):
            results = self.search(item.question, history.store_key)
        seconds = time.perf_counter() - start
        kept = fill_to_budget([r["memory"] for r in results], int(self.retrieval["token_budget"]))
        text = "\n".join(f"- {m}" for m in kept)
        return ArmContext(
            text=text,
            context_tokens=count_tokens(text),
            cache_prefix=False,
            retrieved=[str(r.get("id", "")) for r in results[: len(kept)]],
            candidates=len(results),
            retrieval_seconds=seconds,
            retrieval_cost=self.query_sink.since(before),
            stored_text=self.stored_text(history.store_key),
        )
