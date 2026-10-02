"""Arm C: plain RAG over verbatim chunks. The control arm: no extraction, no memory logic."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from memstudy.arms.base import ArmContext, IngestStats, fill_to_budget
from memstudy.chunking import chunk_document, split_oversized
from memstudy.metering import CostSink, Meter
from memstudy.schema import History, Item, Session, render_session
from memstudy.tokens import count_tokens

EMBED_BATCH = 256


def chunk_session(session: Session, chunk_tokens: int) -> list[str]:
    """Verbatim chunks of whole turns, each starting with the session header."""
    header = render_session(Session(session_id=session.session_id, timestamp=session.timestamp, turns=[]))
    lines: list[str] = []
    for turn in session.turns:
        line = f"{turn.speaker}: {turn.text}"
        if count_tokens(line) > chunk_tokens:
            lines.extend(split_oversized(line, chunk_tokens))
        else:
            lines.append(line)
    chunks: list[str] = []
    buffer: list[str] = []
    size = count_tokens(header)
    for line in lines:
        n = count_tokens(line)
        if buffer and size + n > chunk_tokens:
            chunks.append("\n".join([header, *buffer]))
            buffer, size = [], count_tokens(header)
        buffer.append(line)
        size += n
    if buffer:
        chunks.append("\n".join([header, *buffer]))
    return chunks


def chunk_history(history: History, chunk_tokens: int) -> list[str]:
    if history.document is not None:
        return chunk_document(history.document, chunk_tokens)
    return [c for s in history.sessions for c in chunk_session(s, chunk_tokens)]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.asarray(vectors / np.clip(norms, 1e-12, None), dtype=np.float32)


class RagArm:
    name = "C"

    def __init__(
        self,
        cfg: dict[str, Any],
        retrieval: dict[str, Any],
        client: Any,
        meter: Meter,
        cache_dir: Path,
    ) -> None:
        self.chunk_tokens: int = cfg["chunk_tokens"]
        self.pool: int = retrieval["candidate_pool"]
        self.budget: int = retrieval["token_budget"]
        self.embed_model: str = cfg["embedding_model"]
        self.client = client
        self.meter = meter
        self.cache_dir = Path(cache_dir) / self.embed_model / str(self.chunk_tokens)
        self.ingest_sink = CostSink()
        self.query_sink = CostSink()
        self._index: dict[str, tuple[list[str], np.ndarray]] = {}

    def _embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i : i + EMBED_BATCH]
            resp = self.client.embeddings.create(model=self.embed_model, input=batch)
            out.extend(d.embedding for d in resp.data)
        return _normalize(np.asarray(out, dtype=np.float32))

    def prepare(self, history: History) -> IngestStats | None:
        if history.history_id in self._index:
            return None
        chunks = chunk_history(history, self.chunk_tokens)
        digest = hashlib.sha256("\x1f".join(chunks).encode()).hexdigest()
        path = self.cache_dir / f"{history.user_id}.npz"
        before = self.ingest_sink.snapshot()
        start = time.perf_counter()
        vectors: np.ndarray | None = None
        if path.exists():
            data = np.load(path)
            if str(data["digest"]) == digest:
                vectors = data["vectors"]
        if vectors is None:
            with self.meter.use_sink(self.ingest_sink):
                vectors = self._embed(chunks)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, vectors=vectors, digest=np.asarray(digest))
        self._index[history.history_id] = (chunks, vectors)
        return IngestStats(
            seconds=time.perf_counter() - start,
            cost=self.ingest_sink.since(before),
            stored_units=len(chunks),
        )

    def context(self, item: Item, history: History) -> ArmContext:
        self.prepare(history)
        chunks, vectors = self._index[history.history_id]
        before = self.query_sink.snapshot()
        start = time.perf_counter()
        with self.meter.use_sink(self.query_sink):
            query = self._embed([item.question])[0]
        scores = vectors @ query
        top = np.argsort(-scores)[: min(self.pool, len(chunks))]
        kept = fill_to_budget([chunks[i] for i in top], self.budget)
        text = "\n\n---\n\n".join(kept)
        return ArmContext(
            text=text,
            context_tokens=count_tokens(text),
            cache_prefix=False,
            retrieved=[str(int(i)) for i in top[: len(kept)]],
            candidates=len(top),
            retrieval_seconds=time.perf_counter() - start,
            retrieval_cost=self.query_sink.since(before),
        )
