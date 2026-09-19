import asyncio

import httpx


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

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
        payload = {"model": self.model, "input": texts, "encoding_format": "float"}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(
                        f"{self.base_url}/embeddings", json=payload, headers=headers
                    )
                    response.raise_for_status()
                    data = sorted(response.json()["data"], key=lambda item: item["index"])
                    vectors = [item["embedding"] for item in data]
                    if len(vectors) != len(texts):
                        raise ValueError("Embedding response length did not match input length")
                    return vectors
                except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                    last_error = exc
                    if attempt + 1 < self.max_retries:
                        await asyncio.sleep(0.25 * (2**attempt))
        raise RuntimeError("Embedding request failed after retries") from last_error
