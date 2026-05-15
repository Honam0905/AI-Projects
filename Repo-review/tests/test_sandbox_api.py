"""API tests for the sandbox service."""

from pathlib import Path

from fastapi.testclient import TestClient

from agent_swarms.api.app import create_app
from agent_swarms.api.dependencies import (
    get_review_event_broker,
    get_review_run_store,
    get_sandbox_service,
)
from agent_swarms.config import get_settings
from agent_swarms.graphs.compile_graph import get_compiled_graph


def _build_client(monkeypatch, tmp_path: Path) -> TestClient:
    monkeypatch.setenv("AGENT_SWARMS_LLM_BACKEND", "mock")
    monkeypatch.setenv("AGENT_SWARMS_SANDBOX_BACKEND", "local")
    monkeypatch.setenv("AGENT_SWARMS_SANDBOX_DATA_DIR", str(tmp_path / ".agent_swarms_data"))
    monkeypatch.setenv("AGENT_SWARMS_LOCAL_SANDBOX_USE_MACOS_PROFILE", "false")
    get_settings.cache_clear()
    get_compiled_graph.cache_clear()
    get_review_event_broker.cache_clear()
    get_review_run_store.cache_clear()
    get_sandbox_service.cache_clear()
    return TestClient(create_app())


def test_create_sandbox_run_and_list_files(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "hello.txt").write_text("hello sandbox")

    client = _build_client(monkeypatch, tmp_path)

    create_response = client.post("/api/sandbox/runs", json={"repo_path": str(repo_path)})

    assert create_response.status_code == 201
    payload = create_response.json()
    run_id = payload["run_id"]
    assert payload["sandbox_backend"] == "local"
    assert payload["status"] == "ready"

    list_response = client.get(f"/api/sandbox/runs/{run_id}/files", params={"path": "."})

    assert list_response.status_code == 200
    file_names = [entry["path"] for entry in list_response.json()["entries"]]
    assert "hello.txt" in file_names


def test_shell_and_python_execution_persist_artifacts(monkeypatch, tmp_path: Path) -> None:
    client = _build_client(monkeypatch, tmp_path)
    create_response = client.post("/api/sandbox/runs", json={})
    run_id = create_response.json()["run_id"]

    shell_response = client.post(
        f"/api/sandbox/runs/{run_id}/shell",
        json={"command": "printf 'sandbox-shell'"},
    )

    assert shell_response.status_code == 200
    shell_payload = shell_response.json()
    assert shell_payload["exit_code"] == 0
    assert shell_payload["stdout"] == "sandbox-shell"

    python_response = client.post(
        f"/api/sandbox/runs/{run_id}/python",
        json={"code": "print('sandbox-python')"},
    )

    assert python_response.status_code == 200
    python_payload = python_response.json()
    assert python_payload["exit_code"] == 0
    assert python_payload["stdout"] == "sandbox-python\n"

    artifacts_response = client.get(f"/api/sandbox/runs/{run_id}/artifacts")

    assert artifacts_response.status_code == 200
    assert len(artifacts_response.json()["artifacts"]) == 2


def test_shell_execution_blocks_host_path_access(monkeypatch, tmp_path: Path) -> None:
    client = _build_client(monkeypatch, tmp_path)
    create_response = client.post("/api/sandbox/runs", json={})
    run_id = create_response.json()["run_id"]

    shell_response = client.post(
        f"/api/sandbox/runs/{run_id}/shell",
        json={"command": "cat /etc/hosts"},
    )

    assert shell_response.status_code == 400
    assert "outside the sandbox workspace" in shell_response.json()["detail"]


def test_python_execution_blocks_host_file_access(monkeypatch, tmp_path: Path) -> None:
    client = _build_client(monkeypatch, tmp_path)
    create_response = client.post("/api/sandbox/runs", json={})
    run_id = create_response.json()["run_id"]

    python_response = client.post(
        f"/api/sandbox/runs/{run_id}/python",
        json={"code": "print(open('/etc/hosts').read())"},
    )

    assert python_response.status_code == 200
    payload = python_response.json()
    assert payload["exit_code"] != 0
    assert "escapes the local sandbox workspace" in payload["stderr"]


def test_read_file_and_close_run(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "notes.md").write_text("sandbox notes")

    client = _build_client(monkeypatch, tmp_path)
    create_response = client.post("/api/sandbox/runs", json={"repo_path": str(repo_path)})
    run_id = create_response.json()["run_id"]

    read_response = client.get(
        f"/api/sandbox/runs/{run_id}/files/content",
        params={"path": "notes.md"},
    )

    assert read_response.status_code == 200
    assert read_response.json()["content"] == "sandbox notes"

    close_response = client.delete(f"/api/sandbox/runs/{run_id}")

    assert close_response.status_code == 200
    assert close_response.json()["status"] == "closed"


def test_create_sandbox_run_ignores_runtime_cache_directories(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "README.md").write_text("# Demo\n")
    runtime_dir = repo_path / ".agent_swarms_data" / "runs" / "old"
    uv_cache_dir = repo_path / ".uv-cache"
    runtime_dir.mkdir(parents=True)
    uv_cache_dir.mkdir()
    (runtime_dir / "artifact.json").write_text("{}")
    (uv_cache_dir / "cache.txt").write_text("cache")

    client = _build_client(monkeypatch, tmp_path)
    create_response = client.post("/api/sandbox/runs", json={"repo_path": str(repo_path)})

    assert create_response.status_code == 201
    run_id = create_response.json()["run_id"]

    list_response = client.get(f"/api/sandbox/runs/{run_id}/files", params={"path": "."})

    assert list_response.status_code == 200
    file_names = [entry["path"] for entry in list_response.json()["entries"]]
    assert "README.md" in file_names
    assert ".agent_swarms_data" not in file_names
    assert ".uv-cache" not in file_names
