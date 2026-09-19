from scopegraph.backends._baseline import (
    BaselineKind,
    BaselineMemorySystem,
    BaselineRepository,
)
from scopegraph.embeddings.base import EmbeddingProvider
from scopegraph.llm.extraction import CandidateExtractor
from scopegraph.memory.promoter import PromotionPolicy
from scopegraph.memory.retriever import RetrievalConfig


class TwoLevelGraphMemory(BaselineMemorySystem):
    """Hybrid graph memory with current-session and global levels only."""

    def __init__(
        self,
        repository: BaselineRepository,
        extractor: CandidateExtractor,
        embedder: EmbeddingProvider,
        *,
        promotion_policy: PromotionPolicy | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        super().__init__(
            repository,
            extractor,
            embedder,
            kind=BaselineKind.TWO_LEVEL_GRAPH,
            promotion_policy=promotion_policy,
            retrieval_config=retrieval_config,
        )


__all__ = ["TwoLevelGraphMemory"]
