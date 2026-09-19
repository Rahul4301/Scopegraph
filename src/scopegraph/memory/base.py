from abc import ABC, abstractmethod
from datetime import datetime

from scopegraph.models.correction import CorrectionRequest, CorrectionResult
from scopegraph.models.retrieval import IngestResult, MemoryStats, RetrievalResult
from scopegraph.models.scope import ScopeRef
from scopegraph.models.session import SessionInput


class MemorySystem(ABC):
    """Fair-comparison contract shared by every experimental memory backend."""

    @abstractmethod
    async def reset(self) -> None: ...

    @abstractmethod
    async def ingest_session(
        self, session: SessionInput, *, current_scope: ScopeRef | None
    ) -> IngestResult: ...

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        *,
        current_scope: ScopeRef | None,
        top_k: int,
        token_budget: int,
        now: datetime | None = None,
    ) -> RetrievalResult: ...

    @abstractmethod
    async def apply_correction(self, correction: CorrectionRequest) -> CorrectionResult: ...

    @abstractmethod
    async def stats(self) -> MemoryStats: ...

