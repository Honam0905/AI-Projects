"""End-to-end review checks across local, Docker, and live GitHub paths."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
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


def _build_client(monkeypatch, tmp_path: Path, *, sandbox_backend: str) -> TestClient:
    monkeypatch.setenv("AGENT_SWARMS_LLM_BACKEND", "mock")
    monkeypatch.setenv("AGENT_SWARMS_SANDBOX_BACKEND", sandbox_backend)
    monkeypatch.setenv("AGENT_SWARMS_SANDBOX_DATA_DIR", str(tmp_path / ".agent_swarms_data"))
    monkeypatch.setenv("AGENT_SWARMS_LOCAL_SANDBOX_USE_MACOS_PROFILE", "false")
    get_settings.cache_clear()
    get_compiled_graph.cache_clear()
    get_review_event_broker.cache_clear()
    get_review_run_store.cache_clear()
    get_sandbox_service.cache_clear()
    get_review_service.cache_clear()
    return TestClient(create_app())


def _docker_available() -> bool:
    docker = shutil.which("docker")
    if docker is None:
        return False
    try:
        subprocess.run(
            [docker, "info"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return False
    return True


def test_e2e_local_backend_maps_repo(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "README.md").write_text("# Demo\n\nInstall\nSetup\nRun\nTest\n")
    (repo_path / "src").mkdir()
    (repo_path / "src" / "app.py").write_text("print('demo')\n")
    (repo_path / "tests").mkdir()
    (repo_path / "tests" / "test_app.py").write_text("def test_app():\n    assert True\n")

    client = _build_client(monkeypatch, tmp_path, sandbox_backend="local")
    response = client.post(
        "/api/review",
        json={
            "repo_path": str(repo_path),
            "user_query": "Please review docs and onboarding for this repository.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["report"]["repo_map"]["mapped_file_count"] >= 3
    assert "./README.md" in payload["report"]["repo_map"]["docs_files"]
    assert "./tests/test_app.py" in payload["report"]["repo_map"]["test_locations"]


@pytest.mark.skipif(
    not _docker_available(),
    reason="Docker is not available for real end-to-end checks",
)
def test_e2e_docker_backend_maps_repo(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "README.md").write_text("# Demo\n\nInstall\nSetup\nRun\nTest\n")
    (repo_path / "main.py").write_text("print('demo')\n")

    client = _build_client(monkeypatch, tmp_path, sandbox_backend="docker")
    response = client.post(
        "/api/review",
        json={
            "repo_path": str(repo_path),
            "sandbox_backend": "docker",
            "user_query": "Please review docs and onboarding for this repository.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["sandbox_backend"] == "docker"
    assert payload["report"]["repo_map"]["mapped_file_count"] >= 2
    assert "./README.md" in payload["report"]["repo_map"]["docs_files"]


@pytest.mark.skipif(
    not os.getenv("AGENT_SWARMS_E2E_GITHUB_REPO_URL"),
    reason="Set AGENT_SWARMS_E2E_GITHUB_REPO_URL to run the live GitHub clone check",
)
def test_e2e_live_github_remote_clone_path(monkeypatch, tmp_path: Path) -> None:
    client = _build_client(monkeypatch, tmp_path, sandbox_backend="local")
    response = client.post(
        "/api/review",
        json={
            "repo_url": os.environ["AGENT_SWARMS_E2E_GITHUB_REPO_URL"],
            "user_query": "Please review docs and onboarding for this repository.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["repo_source_type"] == "remote"
    assert payload["report"]["repo_map"]["mapped_file_count"] > 0
