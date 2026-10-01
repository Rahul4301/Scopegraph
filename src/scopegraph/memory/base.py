from abc import ABC, abstractmethod
from datetime import datetime

from scopegraph.models.correction import CorrectionRequest, CorrectionResult
from scopegraph.models.retrieval import IngestResult, MemoryStats, RetrievalResult
from scopegraph.models.scope import ScopeRef
from scopegraph.models.session import SessionInput


class MemorySystem(ABC):
    """Contract implemented by the ScopeGraph memory service."""

    @abstractmethod
    async def reset(self) -> None:
        """Remove all stored memory (isolated experiment namespaces only)."""

    @abstractmethod
    async def ingest_session(
        self, session: SessionInput, *, current_scope: ScopeRef | None
    ) -> IngestResult:
        """Store a session's messages and consolidate memories from them."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        *,
        current_scope: ScopeRef | None,
        top_k: int,
        token_budget: int,
        now: datetime | None = None,
    ) -> RetrievalResult:
        """Retrieve scoped evidence for a query."""

    @abstractmethod
    async def apply_correction(self, correction: CorrectionRequest) -> CorrectionResult:
        """Apply an audited correction."""

    @abstractmethod
    async def stats(self) -> MemoryStats:
        """Return logical node and relationship counts."""
