from datetime import datetime

from fastapi.testclient import TestClient

from scopegraph.api.dependencies import get_memory_system, get_repository
from scopegraph.api.main import app
from scopegraph.graph.in_memory import InMemoryMemoryRepository
from scopegraph.models.retrieval import RetrievalResult
from scopegraph.models.scope import ScopeRef


class FakeRetrievalSystem:
    def __init__(self) -> None:
        self.current_scope: ScopeRef | None = None

    async def retrieve(
        self,
        query: str,
        *,
        current_scope: ScopeRef | None,
        top_k: int,
        token_budget: int,
        now: datetime | None = None,
    ) -> RetrievalResult:
        self.current_scope = current_scope
        return RetrievalResult(
            items=[],
            retrieval_latency_ms=0,
            token_count=0,
            trace=[],
            backend_name="scopegraph",
        )


def test_scope_crud_api() -> None:
    repository = InMemoryMemoryRepository()
    app.dependency_overrides[get_repository] = lambda: repository
    try:
        with TestClient(app) as client:
            created = client.post(
                "/scopes",
                json={"id": "global", "name": "Global", "scope_type": "global"},
            )
            assert created.status_code == 201
            assert created.json()["id"] == "global"

            listed = client.get("/scopes")
            assert listed.status_code == 200
            assert [item["id"] for item in listed.json()] == ["global"]

            updated = client.patch("/scopes/global", json={"name": "Global memory"})
            assert updated.status_code == 200
            assert updated.json()["name"] == "Global memory"
    finally:
        app.dependency_overrides.clear()


def test_retrieval_api() -> None:
    memory_system = FakeRetrievalSystem()
    app.dependency_overrides[get_memory_system] = lambda: memory_system
    try:
        with TestClient(app) as client:
            response = client.post(
                "/retrieve",
                json={
                    "query": "What database does Beta use?",
                    "current_scope": {
                        "id": "beta",
                        "name": "Beta",
                        "session_id": "session-1",
                    },
                    "top_k": 4,
                    "token_budget": 100,
                },
            )
        assert response.status_code == 200
        assert response.json()["backend_name"] == "scopegraph"
        assert memory_system.current_scope is not None
        assert memory_system.current_scope.session_id == "session-1"
    finally:
        app.dependency_overrides.clear()
