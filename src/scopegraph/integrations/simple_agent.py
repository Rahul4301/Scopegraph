"""Small framework-independent agent demonstrating the MemorySystem contract."""

from datetime import datetime

from scopegraph.llm.answering import AnswerModel
from scopegraph.memory.base import MemorySystem
from scopegraph.models.retrieval import IngestResult, RetrievalResult
from scopegraph.models.scope import ScopeRef
from scopegraph.models.session import SessionInput


class SimpleMemoryAgent:
    def __init__(self, memory: MemorySystem, answerer: AnswerModel | None = None) -> None:
        self.memory = memory
        self.answerer = answerer

    async def remember(
        self, session: SessionInput, *, current_scope: ScopeRef | None
    ) -> IngestResult:
        return await self.memory.ingest_session(session, current_scope=current_scope)

    async def ask(
        self,
        question: str,
        *,
        current_scope: ScopeRef | None,
        top_k: int = 8,
        token_budget: int = 1500,
        now: datetime | None = None,
    ) -> tuple[str | None, RetrievalResult]:
        result = await self.memory.retrieve(
            question, current_scope=current_scope, top_k=top_k,
            token_budget=token_budget, now=now,
        )
        answer = (
            await self.answerer.generate(question=question, context=result.items)
            if self.answerer is not None else (result.items[0].content if result.items else None)
        )
        return answer, result
