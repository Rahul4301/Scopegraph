from fastapi.testclient import TestClient

from scopegraph.api.dependencies import get_repository
from scopegraph.api.main import app
from scopegraph.graph.in_memory import InMemoryMemoryRepository


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

