"""Optional dependency-free adapter shape for LangGraph-style state hooks."""

from typing import Any

from scopegraph.integrations.simple_agent import SimpleMemoryAgent
from scopegraph.models.scope import ScopeRef
from scopegraph.models.session import SessionInput


class ScopeGraphLangGraphAdapter:
    """Bridge memory calls without making LangGraph a core dependency."""

    def __init__(self, agent: SimpleMemoryAgent) -> None:
        self.agent = agent

    async def ingest_state(
        self, state: dict[str, Any], *, current_scope: ScopeRef | None
    ) -> dict[str, Any]:
        session = state.get("scopegraph_session")
        if isinstance(session, SessionInput):
            result = await self.agent.remember(session, current_scope=current_scope)
            state["scopegraph_ingest"] = result.model_dump(mode="json")
        return state

    async def retrieve_state(
        self, state: dict[str, Any], *, current_scope: ScopeRef | None
    ) -> dict[str, Any]:
        question = state.get("question")
        if isinstance(question, str) and question:
            answer, result = await self.agent.ask(question, current_scope=current_scope)
            state["answer"] = answer
            state["scopegraph_retrieval"] = result.model_dump(mode="json")
        return state
