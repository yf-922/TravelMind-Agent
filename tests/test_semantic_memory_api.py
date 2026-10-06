from fastapi.testclient import TestClient

from app.core.auth import create_token
from app.main import app


def test_semantic_memory_delete_requires_authentication():
    response = TestClient(app).delete("/api/profile/semantic-memory")
    assert response.status_code == 401


def test_semantic_memory_delete_returns_provider_failure(monkeypatch):
    import app.api.profile_routes as routes

    monkeypatch.setattr(routes, "delete_user_memories", lambda user_id: (False, "degraded"))
    response = TestClient(app).delete(
        "/api/profile/semantic-memory",
        headers={"Authorization": "Bearer " + create_token("memory-owner")},
    )
    assert response.status_code == 503


def test_semantic_memory_delete_uses_authenticated_owner(monkeypatch):
    import app.api.profile_routes as routes
    seen = []

    monkeypatch.setattr(routes, "delete_user_memories", lambda user_id: seen.append(user_id) or (True, "ok"))
    response = TestClient(app).delete(
        "/api/profile/semantic-memory",
        headers={"Authorization": "Bearer " + create_token("memory-owner")},
    )
    assert response.status_code == 200
    assert response.json() == {"deleted": True, "status": "ok"}
    assert seen == ["memory-owner"]
