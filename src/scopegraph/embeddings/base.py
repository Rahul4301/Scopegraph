from typing import Protocol


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str:
        """Name of the embedding model."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed each text into a vector."""
