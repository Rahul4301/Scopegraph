from scopegraph.backends._baseline import (
    BaselineKind,
    BaselineMemorySystem,
    BaselineRepository,
)
from scopegraph.embeddings.base import EmbeddingProvider
from scopegraph.llm.extraction import CandidateExtractor
from scopegraph.memory.retriever import RetrievalConfig


class VectorMemory(BaselineMemorySystem):
    """Flat semantic memory with no graph traversal or scope validity."""

    def __init__(
        self,
        repository: BaselineRepository,
        extractor: CandidateExtractor,
        embedder: EmbeddingProvider,
        *,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        super().__init__(
            repository,
            extractor,
            embedder,
            kind=BaselineKind.VECTOR,
            retrieval_config=retrieval_config,
        )


__all__ = ["VectorMemory"]
