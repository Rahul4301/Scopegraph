"""Credential-free deterministic embeddings for local demos and smoke tests."""


class HashEmbeddingProvider:
    """Small hash-bucket vectors, intentionally not a quality semantic model."""

    model_name = "local-hash-v1"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            buckets = [0.0] * 8
            for index, char in enumerate(text.lower()):
                buckets[(ord(char) + index) % len(buckets)] += 1.0
            magnitude = sum(value * value for value in buckets) ** 0.5 or 1.0
            vectors.append([value / magnitude for value in buckets])
        return vectors
