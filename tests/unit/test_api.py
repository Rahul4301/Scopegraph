import asyncio
from datetime import datetime

from fastapi.testclient import TestClient

from scopegraph.api.dependencies import (
    get_correction_service,
    get_memory_system,
    get_repository,
)
from scopegraph.api.main import app
from scopegraph.graph.in_memory import InMemoryMemoryRepository
from scopegraph.memory.corrections import CorrectionService
from scopegraph.models.memory import MemoryCreate, MemoryType, ScopeLevel
from scopegraph.models.retrieval import RetrievalResult
from scopegraph.models.scope import ScopeCreate, ScopeRef, ScopeType
from scopegraph.models.session import SessionCreate
from scopegraph.models.source import MessageRole, SourceMessageCreate


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


def test_correction_api_round_trip() -> None:
    repository = InMemoryMemoryRepository()

    async def seed() -> None:
        await repository.create_scope(
            ScopeCreate(id="global", name="Global", scope_type=ScopeType.GLOBAL)
        )
        await repository.create_memory(
            MemoryCreate(
                id="memory-1",
                content="Old content",
                memory_type=MemoryType.FACT,
                scope_level=ScopeLevel.GLOBAL,
                scope_id="global",
            )
        )

    asyncio.run(seed())
    corrections = CorrectionService(repository)
    app.dependency_overrides[get_repository] = lambda: repository
    app.dependency_overrides[get_correction_service] = lambda: corrections
    try:
        with TestClient(app) as client:
            edited = client.patch(
                "/memories/memory-1",
                json={"content": "Correct content", "reason": "fix typo"},
            )
            assert edited.status_code == 200
            assert edited.json()["memory"]["revision"] == 2

            preview = client.post("/memories/memory-1/prune/preview")
            assert preview.status_code == 200
            assert preview.json()["hard_delete"] is False

            pruned = client.post(
                "/memories/memory-1/prune", json={"reason": "bad evidence"}
            )
            assert pruned.status_code == 200
            prune_event_id = pruned.json()["event"]["id"]

            restored = client.post(
                "/memories/memory-1/restore",
                json={"undo_of": prune_event_id, "reason": "undo"},
            )
            assert restored.status_code == 200
            assert restored.json()["memory"]["status"] == "active"

            history = client.get("/memories/memory-1/history")
            assert history.status_code == 200
            assert [event["action"] for event in history.json()] == [
                "edit",
                "tombstone",
                "restore",
            ]
    finally:
        app.dependency_overrides.clear()


def test_graph_provenance_export_and_stats_api() -> None:
    repository = InMemoryMemoryRepository()

    async def seed() -> None:
        await repository.create_scope(
            ScopeCreate(id="global", name="Global", scope_type=ScopeType.GLOBAL)
        )
        await repository.create_session(SessionCreate(id="session-1", scope_id="global"))
        await repository.create_source_message(
            SourceMessageCreate(
                id="source-1",
                session_id="session-1",
                role=MessageRole.USER,
                content="The user prefers PostgreSQL",
                turn_index=0,
            )
        )
        await repository.create_memory(
            MemoryCreate(
                id="memory-1",
                content="User prefers PostgreSQL",
                memory_type=MemoryType.PREFERENCE,
                scope_level=ScopeLevel.GLOBAL,
                scope_id="global",
                source_ids=["source-1"],
            )
        )

    asyncio.run(seed())
    app.dependency_overrides[get_repository] = lambda: repository
    try:
        with TestClient(app) as client:
            graph = client.get("/graph/subgraph?include_sources=true")
            assert graph.status_code == 200
            assert {node["id"] for node in graph.json()["nodes"]} == {
                "global",
                "memory-1",
                "source-1",
            }

            provenance = client.get("/memories/memory-1/provenance")
            assert provenance.status_code == 200
            assert provenance.json()["source_messages"][0]["id"] == "source-1"

            exported = client.get("/graph/export?format=graphml")
            assert exported.status_code == 200
            assert "<graphml" in exported.text
            assert "memory-1" in exported.text

            stats = client.get("/stats")
            assert stats.status_code == 200
            assert stats.json()["memory_count"] == 1
    finally:
        app.dependency_overrides.clear()
