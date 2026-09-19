import hashlib
import json
import sqlite3
from pathlib import Path

from scopegraph.embeddings.base import EmbeddingProvider


def content_hash(model_name: str, text: str) -> str:
    payload = f"{model_name}\0{text}".encode()
    return hashlib.sha256(payload).hexdigest()


class SQLiteEmbeddingCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                content_hash TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                vector_json TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def get(self, key: str) -> list[float] | None:
        row = self._connection.execute(
            "SELECT vector_json FROM embeddings WHERE content_hash = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, model_name: str, vector: list[float]) -> None:
        self._connection.execute(
            """
            INSERT OR REPLACE INTO embeddings(content_hash, model_name, vector_json)
            VALUES (?, ?, ?)
            """,
            (key, model_name, json.dumps(vector, separators=(",", ":"))),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()


class CachedEmbedder:
    def __init__(self, provider: EmbeddingProvider, cache: SQLiteEmbeddingCache) -> None:
        self.provider = provider
        self.cache = cache

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float] | None] = []
        missing_texts: list[str] = []
        missing_indexes: list[int] = []
        keys = [content_hash(self.model_name, text) for text in texts]
        for index, key in enumerate(keys):
            cached = self.cache.get(key)
            vectors.append(cached)
            if cached is None:
                missing_indexes.append(index)
                missing_texts.append(texts[index])
        if missing_texts:
            generated = await self.provider.embed(missing_texts)
            if len(generated) != len(missing_texts):
                raise ValueError("Embedding provider returned the wrong number of vectors")
            for index, vector in zip(missing_indexes, generated, strict=True):
                vectors[index] = vector
                self.cache.put(keys[index], self.model_name, vector)
        if any(vector is None for vector in vectors):
            raise RuntimeError("Embedding cache did not resolve every requested vector")
        return [vector for vector in vectors if vector is not None]
