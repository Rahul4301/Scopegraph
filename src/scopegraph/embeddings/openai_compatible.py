import hashlib
import math

from scopegraph.llm.transport import ModelTransport


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        batch_size: int = 128,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        if batch_size < 1:
            raise ValueError("Embedding batch size must be positive")
        self.batch_size = batch_size
        self.transport = ModelTransport(timeout=timeout_seconds, retries=max_retries)

    @property
    def cache_namespace(self) -> str:
        endpoint = hashlib.sha256(self.base_url.encode()).hexdigest()[:16]
        return f"{endpoint}:{self.model}"

    async def aclose(self) -> None:
        await self.transport.aclose()

    @property
    def model_name(self) -> str:
        return self.model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key or not self.model:
            raise RuntimeError(
                "EMBEDDING_API_KEY and EMBEDDING_MODEL are required for live retrieval"
            )
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), self.batch_size):
            batch = texts[offset:offset + self.batch_size]
            body = await self.transport.post(
                f"{self.base_url}/embeddings", api_key=self.api_key,
                payload={"model": self.model, "input": batch, "encoding_format": "float"},
            )
            data = sorted(body["data"], key=lambda item: item["index"])
            if [item["index"] for item in data] != list(range(len(batch))):
                raise ValueError("Embedding response indexes do not match input")
            for item in data:
                vector = item["embedding"]
                if not vector or not all(isinstance(value, int | float) and math.isfinite(value)
                                         for value in vector):
                    raise ValueError("Embedding response contains an invalid vector")
                if vectors and len(vector) != len(vectors[0]):
                    raise ValueError("Embedding dimensions changed within a response")
                vectors.append(vector)
        return vectors
