"""API tests for the supervisor chat backend."""

from fastapi.testclient import TestClient

from agent_swarms.api.app import create_app
from agent_swarms.api.dependencies import (
    get_review_event_broker,
    get_review_run_store,
    get_review_service,
    get_sandbox_service,
)
from agent_swarms.config import get_settings
from agent_swarms.graphs.compile_graph import get_compiled_graph


def _build_client(monkeypatch) -> TestClient:
    monkeypatch.setenv("AGENT_SWARMS_LLM_BACKEND", "mock")
    get_settings.cache_clear()
    get_compiled_graph.cache_clear()
    get_review_event_broker.cache_clear()
    get_review_run_store.cache_clear()
    get_sandbox_service.cache_clear()
    get_review_service.cache_clear()
    return TestClient(create_app())


def test_health_check(monkeypatch) -> None:
    client = _build_client(monkeypatch)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "environment": "development",
        "llm_backend": "mock",
    }


def test_chat_returns_supervisor_response(monkeypatch) -> None:
    client = _build_client(monkeypatch)

    response = client.post("/api/chat", json={"message": "hello there"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["provider"] == "mock"
    assert payload["plan"]["mode"] == "chat"
    assert payload["plan"]["spawn_count"] == 0
    assert "supervisor agent" in payload["response"].lower()


def test_mock_chat_handles_name_and_model_questions(monkeypatch) -> None:
    client = _build_client(monkeypatch)

    name_response = client.post("/api/chat", json={"message": "What is your name?"})
    model_response = client.post("/api/chat", json={"message": "Which model are you?"})

    assert name_response.status_code == 200
    assert "supervisor agent" in name_response.json()["response"].lower()

    assert model_response.status_code == 200
    assert "mock supervisor backend" in model_response.json()["response"].lower()


def test_review_intent_is_detected_without_spawning(monkeypatch) -> None:
    client = _build_client(monkeypatch)

    response = client.post(
        "/api/chat",
        json={"message": "Can you review this repo and tell me if it is production-ready?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["mode"] == "review"
    assert payload["plan"]["spawn_count"] == 6
    assert payload["plan"]["needs_sandbox"] is True
    assert payload["plan"]["selected_workers"] == [
        "repo_mapper",
        "static_reviewer",
        "runtime_tester",
        "security_reviewer",
        "docs_devex_reviewer",
        "external_validator",
    ]
    assert "review swarm" in payload["response"].lower()
    assert "repo target" in payload["response"].lower()


def test_runtime_review_intent_gets_targeted_worker_plan(monkeypatch) -> None:
    client = _build_client(monkeypatch)

    response = client.post(
        "/api/chat",
        json={"message": "Please run a test and build focused review for this codebase."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["mode"] == "review"
    assert payload["plan"]["spawn_count"] == 2
    assert payload["plan"]["selected_workers"] == ["repo_mapper", "runtime_tester"]


def test_requested_review_mode_forces_review_path(monkeypatch) -> None:
    client = _build_client(monkeypatch)

    response = client.post(
        "/api/chat",
        json={
            "message": "Please help with this repository.",
            "requested_mode": "review",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["mode"] == "review"
    assert payload["plan"]["spawn_count"] >= 1
