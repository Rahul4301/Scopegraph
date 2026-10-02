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


def session_messages(session: Session, speakers: dict[str, str]) -> list[dict[str, str]]:
    """Messages for one session. LoCoMo speakers are mapped to user/assistant by first
    appearance; LongMemEval roles are kept. Each message carries the session date."""
    stamp = f"[{session.timestamp}] " if session.timestamp else ""
    out: list[dict[str, str]] = []
    for turn in session.turns:
        if turn.speaker in ("user", "assistant"):
            role, content = turn.speaker, turn.text
        else:
            role = speakers.setdefault(turn.speaker, "user" if not speakers else "assistant")
            content = f"{turn.speaker}: {turn.text}"
        out.append({"role": role, "content": f"{stamp}{content}"})
    return out


def build_memory_config(cfg: dict[str, Any], vector_path: str, history_db: str) -> dict[str, Any]:
    return {
        "history_db_path": history_db,
        "llm": {"provider": "openai", "config": {"model": cfg["extraction_model"]}},
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

    def stored_count(self, user_id: str) -> int:
        found = self.memory.get_all(filters={"user_id": user_id}, top_k=100_000)
        return len(found["results"])

    def prepare(self, history: History) -> IngestStats | None:
        user_id = history.user_id
        if history.history_id in self._ingested:
            return None
        before = self.ingest_sink.snapshot()
        start = time.perf_counter()
        step = int(self.cfg["turns_per_add"])
        speakers: dict[str, str] = {}
        with self.meter.use_sink(self.ingest_sink):
            for session in history.sessions:
                messages = session_messages(session, speakers)
                for i in range(0, len(messages), step):
                    self.memory.add(
                        messages[i : i + step],
                        user_id=user_id,
                        metadata={"session_id": session.session_id},
                        infer=self.infer,
                    )
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
            results = self.search(item.question, history.user_id)
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
        )
