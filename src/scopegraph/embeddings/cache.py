import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from threading import RLock

from scopegraph.embeddings.base import EmbeddingProvider


def content_hash(model_name: str, text: str) -> str:
    payload = f"{model_name}\0{text}".encode()
    return hashlib.sha256(payload).hexdigest()


class SQLiteEmbeddingCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # FastAPI may execute requests on different worker threads while the
        # cached embedder is shared by the process. Serialize the tiny cache
        # reads/writes instead of leaking sqlite's thread-affinity exception.
        self._lock = RLock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
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
        with self._lock:
            row = self._connection.execute(
                "SELECT vector_json FROM embeddings WHERE content_hash = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        unique = list(dict.fromkeys(keys))
        with self._lock:
            for offset in range(0, len(unique), 800):
                batch = unique[offset:offset + 800]
                placeholders = ",".join("?" for _ in batch)
                rows = self._connection.execute(
                    f"SELECT content_hash, vector_json FROM embeddings "
                    f"WHERE content_hash IN ({placeholders})", batch,
                ).fetchall()
                found.update({key: json.loads(vector) for key, vector in rows})
        return found

    def put_many(self, values: dict[str, list[float]], model_name: str) -> None:
        with self._lock:
            with self._connection:
                self._connection.executemany(
                    "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)",
                    [(key, model_name, json.dumps(vector, separators=(",", ":")))
                     for key, vector in values.items()],
                )

    def put(self, key: str, model_name: str, vector: list[float]) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO embeddings(content_hash, model_name, vector_json)
                VALUES (?, ?, ?)
                """,
                (key, model_name, json.dumps(vector, separators=(",", ":"))),
            )
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class CachedEmbedder:
    def __init__(self, provider: EmbeddingProvider, cache: SQLiteEmbeddingCache) -> None:
        self.provider = provider
        self.cache = cache

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        namespace = str(getattr(self.provider, "cache_namespace", self.model_name))
        keys = [content_hash(namespace, text) for text in texts]
        vectors = await asyncio.to_thread(self.cache.get_many, keys)
        missing_texts = {key: text for key, text in zip(keys, texts, strict=True)
                         if key not in vectors}
        if missing_texts:
            generated = await self.provider.embed(list(missing_texts.values()))
            if len(generated) != len(missing_texts):
                raise ValueError("Embedding provider returned the wrong number of vectors")
            additions = dict(zip(missing_texts, generated, strict=True))
            await asyncio.to_thread(self.cache.put_many, additions, namespace)
            vectors.update(additions)
        return [list(vectors[key]) for key in keys]
