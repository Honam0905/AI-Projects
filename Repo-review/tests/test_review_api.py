"""API tests for the repository review flow."""

import shutil
import subprocess
import time
from datetime import UTC, datetime
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
from agent_swarms.agents.fix_subagent import FixAgentResult, FixChange
from agent_swarms.config import get_settings
from agent_swarms.graphs.compile_graph import get_compiled_graph
from agent_swarms.sandbox.base import SandboxSession
from agent_swarms.sandbox.local_adapter import LocalSandboxAdapter
from agent_swarms.sandbox.service import SandboxService
from agent_swarms.state.enums import ProviderBackend, SandboxBackend, SandboxRunStatus


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
    get_review_service.cache_clear()
    return TestClient(create_app())


def _enable_fake_docker_backend(monkeypatch) -> None:
    original_build_adapter = SandboxService._build_adapter

    class FakeDockerAdapter(LocalSandboxAdapter):
        backend = SandboxBackend.DOCKER

        async def create_session(self, run_id: str) -> SandboxSession:
            return SandboxSession(
                run_id=run_id,
                sandbox_backend=SandboxBackend.DOCKER,
                workspace_path=Path(),
                created_at=datetime.now(UTC),
                status=SandboxRunStatus.READY,
            )

    def fake_build_adapter(self: SandboxService, backend: SandboxBackend):
        if backend is SandboxBackend.DOCKER:
            return FakeDockerAdapter(
                self._workspace_manager,
                use_macos_profile=False,
            )
        return original_build_adapter(self, backend)

    monkeypatch.setattr(SandboxService, "_build_adapter", fake_build_adapter)


def _build_remote_repo_url(tmp_path: Path) -> str:
    if shutil.which("git") is None:
        pytest.skip("git is required for remote repo clone tests")

    source_repo = tmp_path / "source-repo"
    bare_repo = tmp_path / "remote.git"
    source_repo.mkdir()
    (source_repo / "README.md").write_text("# Remote Demo\n\nClone me.\n")
    (source_repo / "app.py").write_text("print('remote review')\n")

    subprocess.run(["git", "init"], cwd=source_repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.com"],
        cwd=source_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Agent Swarms Tests"],
        cwd=source_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "add", "."], cwd=source_repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=source_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "clone", "--bare", str(source_repo), str(bare_repo)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return bare_repo.resolve().as_uri()


def test_review_route_returns_structured_report(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "README.md").write_text("# Demo\n\nRun the project.\n")
    (repo_path / "requirements.txt").write_text("fastapi\npytest==9.0.2\n")
    (repo_path / "app.py").write_text("import os\n\ndef main():\n    eval('1 + 1')\n")

    client = _build_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/review",
        json={
            "repo_path": str(repo_path),
            "user_query": "Is this production-ready? Please do a full repository audit.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["repo_source_type"] == "local"
    assert payload["repo_source"] == str(repo_path)
    assert payload["sandbox_backend"] == "local"
    assert payload["plan"]["spawn_count"] == 6
    assert "repo_mapper" in payload["plan"]["selected_workers"]
    assert payload["report"]["overall_verdict"] in {"caution", "red"}
    assert payload["report"]["repo_map"]["languages"] == ["python"]
    assert payload["report"]["top_findings"]
    assert payload["report"]["artifacts"]


def test_targeted_security_review_uses_narrower_swarm(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "main.py").write_text("import os\nos.system('echo hi')\n")

    client = _build_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/review",
        json={
            "repo_path": str(repo_path),
            "user_query": "Please do a security review of this repo.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plan"]["spawn_count"] == 2
    assert payload["plan"]["selected_workers"] == ["repo_mapper", "security_reviewer"]
    assert any(
        finding["severity"] in {"medium", "high"}
        for finding in payload["report"]["top_findings"]
    )


def test_review_route_supports_remote_repo_url(monkeypatch, tmp_path: Path) -> None:
    repo_url = _build_remote_repo_url(tmp_path)
    client = _build_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/review",
        json={
            "repo_url": repo_url,
            "user_query": "Please review docs and onboarding for this repository.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["repo_source_type"] == "remote"
    assert payload["repo_source"] == repo_url
    assert payload["repo_path"] != repo_url
    assert "./README.md" in payload["report"]["repo_map"]["docs_files"]


def test_review_route_rejects_multi_project_parent_folder(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    first_repo = root / "travel-planner"
    second_repo = root / "crypto-trade-agent"
    first_repo.mkdir(parents=True)
    second_repo.mkdir(parents=True)
    (first_repo / "pyproject.toml").write_text("[project]\nname='travel-planner'\n")
    (second_repo / "package.json").write_text('{"name":"crypto-trade-agent"}\n')

    client = _build_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/review",
        json={
            "repo_path": str(root),
            "user_query": "Please review this repository carefully.",
        },
    )

    assert response.status_code == 400
    assert "multiple repositories" in response.json()["detail"]


def test_review_route_rejects_ambiguous_repo_source(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    client = _build_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/review",
        json={
            "repo_path": str(repo_path),
            "repo_url": "https://github.com/example/repo",
            "user_query": "Please review this repository.",
        },
    )

    assert response.status_code == 422


def test_review_repo_mapper_detects_nested_test_and_docs_paths(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    nested_tests = repo_path / "apps" / "api" / "tests"
    nested_docs = repo_path / "docs" / "guides"
    nested_src = repo_path / "apps" / "api" / "src"
    nested_tests.mkdir(parents=True)
    nested_docs.mkdir(parents=True)
    nested_src.mkdir(parents=True)
    (repo_path / "README.md").write_text("# Demo\n")
    (nested_src / "main.py").write_text("def main():\n    return 'ok'\n")
    (nested_tests / "test_health.py").write_text("def test_health():\n    assert True\n")
    (nested_docs / "setup.md").write_text("# Setup\n")

    client = _build_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/review",
        json={
            "repo_path": str(repo_path),
            "user_query": "Please review this repository carefully.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    repo_map = payload["report"]["repo_map"]
    assert "./apps/api/tests/test_health.py" in repo_map["test_locations"]
    assert "./docs/guides/setup.md" in repo_map["docs_files"]


def test_submit_review_run_and_poll_status(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "README.md").write_text("# Demo\n")
    (repo_path / "main.py").write_text("print('hello')\n")

    client = _build_client(monkeypatch, tmp_path)
    submit_response = client.post(
        "/api/review/runs",
        json={
            "repo_path": str(repo_path),
            "user_query": "Please review this repository.",
        },
    )

    assert submit_response.status_code == 202
    payload = submit_response.json()
    run_id = payload["run_id"]
    assert payload["status"] == "pending"
    assert payload["repo_source_type"] == "local"
    assert payload["repo_source"] == str(repo_path)
    assert payload["sandbox_backend"] == "local"
    assert payload["stream_path"] == f"/api/review/runs/{run_id}/stream"

    final_payload = None
    for _ in range(100):
        detail_response = client.get(f"/api/review/runs/{run_id}")
        assert detail_response.status_code == 200
        final_payload = detail_response.json()
        if final_payload["status"] in {"completed", "failed"}:
            break
        time.sleep(0.02)

    assert final_payload is not None
    assert final_payload["status"] == "completed"
    assert final_payload["repo_source_type"] == "local"
    assert final_payload["repo_source"] == str(repo_path)
    assert final_payload["report"] is not None
    assert final_payload["artifacts"]

    history_response = client.get("/api/review/runs")
    assert history_response.status_code == 200
    assert any(item["run_id"] == run_id for item in history_response.json()["runs"])


def test_review_run_websocket_stream_replays_lifecycle(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "README.md").write_text("# Demo\n")
    (repo_path / "main.py").write_text("import os\nos.system('echo hi')\n")

    client = _build_client(monkeypatch, tmp_path)
    submit_response = client.post(
        "/api/review/runs",
        json={
            "repo_path": str(repo_path),
            "user_query": "Please do a security review of this repo.",
        },
    )
    run_id = submit_response.json()["run_id"]

    event_types: list[str] = []
    with client.websocket_connect(f"/api/review/runs/{run_id}/stream") as websocket:
        while True:
            event = websocket.receive_json()
            event_types.append(event["event_type"])
            if event.get("status") in {"completed", "failed"}:
                break

    assert "run_created" in event_types
    assert "plan_ready" in event_types
    assert "worker_started" in event_types
    assert "tool_completed" in event_types
    assert "run_completed" in event_types


def test_review_route_defaults_to_configured_repo_path(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "README.md").write_text("# Demo\n")
    (repo_path / "main.py").write_text("print('hello')\n")

    monkeypatch.setenv("AGENT_SWARMS_DEFAULT_REVIEW_REPO_PATH", str(repo_path))
    client = _build_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/review",
        json={"user_query": "Please do a security review of this repository."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["plan"]["selected_workers"] == ["repo_mapper", "security_reviewer"]


def test_review_run_tool_events_include_worker_role(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "README.md").write_text("# Demo\n")
    (repo_path / "main.py").write_text("import os\nos.system('echo hi')\n")

    client = _build_client(monkeypatch, tmp_path)
    submit_response = client.post(
        "/api/review/runs",
        json={
            "repo_path": str(repo_path),
            "user_query": "Please do a security review of this repo.",
        },
    )
    run_id = submit_response.json()["run_id"]

    tool_payloads: list[dict] = []
    worker_started_payloads: list[dict] = []
    worker_completed_payloads: list[dict] = []
    with client.websocket_connect(f"/api/review/runs/{run_id}/stream") as websocket:
        while True:
            event = websocket.receive_json()
            if event["event_type"] == "worker_started":
                worker_started_payloads.append(event["payload"])
            if event["event_type"] == "tool_completed":
                tool_payloads.append(event["payload"])
            if event["event_type"] == "worker_completed":
                worker_completed_payloads.append(event["payload"])
            if event.get("status") in {"completed", "failed"}:
                break

    assert worker_started_payloads
    assert any(payload.get("agent_message") for payload in worker_started_payloads)
    assert any(
        payload.get("tool_name") == "security_scan_shell_tool"
        for payload in worker_started_payloads
    )
    assert tool_payloads
    assert any(payload.get("worker_role") == "repo_mapper" for payload in tool_payloads)
    assert any(payload.get("tool_name") == "repo_map_shell_tool" for payload in tool_payloads)
    assert worker_completed_payloads
    assert any(payload.get("agent_message") for payload in worker_completed_payloads)


def test_fix_mode_requires_docker_backend(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "main.py").write_text("print('hello')\n")

    client = _build_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/review",
        json={
            "repo_path": str(repo_path),
            "intent": "fix",
            "sandbox_backend": "local",
            "user_query": "Please fix docs and onboarding issues in this repository.",
        },
    )

    assert response.status_code == 400
    assert "requires sandbox_backend=docker" in response.json()["detail"]


def test_fix_mode_applies_supported_docs_fixes(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "app.py").write_text("print('demo')\n")

    _enable_fake_docker_backend(monkeypatch)
    client = _build_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/review",
        json={
            "repo_path": str(repo_path),
            "intent": "fix",
            "sandbox_backend": "docker",
            "user_query": "Please fix docs and onboarding issues in this repository.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "fix"
    assert payload["sandbox_backend"] == "docker"
    assert payload["fix_result"]["applied"] is True
    assert "README.md" in payload["fix_result"]["changed_files"]
    assert ".env.example" in payload["fix_result"]["changed_files"]
    assert payload["fix_result"]["diff_artifact_path"].endswith(".diff")
    assert payload["fix_result"]["applied_to_local_repo"] is True
    assert payload["fix_result"]["apply_back_target"] == str(repo_path.resolve())
    assert payload["fix_result"]["apply_back_error"] is None
    assert "./README.md" in payload["report"]["repo_map"]["docs_files"]
    assert "README is missing" not in {
        finding["title"] for finding in payload["report"]["top_findings"]
    }
    assert (repo_path / "README.md").exists()
    readme_content = (repo_path / "README.md").read_text()
    assert "## Overview" in readme_content
    assert "python app.py" in readme_content
    assert "cp .env.example .env" in readme_content
    assert "Describe the purpose" not in readme_content
    assert "Document how" not in readme_content
    artifact_response = client.get(
        f"/api/review/runs/{payload['run_id']}/artifacts/content",
        params={"path": payload["fix_result"]["diff_artifact_path"]},
    )
    assert artifact_response.status_code == 200
    assert "README.md" in artifact_response.json()["content"]


def test_fix_mode_improves_existing_readme_when_requested(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "README.md").write_text("# Demo\n\nShort description.\n")
    (repo_path / "app.py").write_text("print('demo')\n")

    _enable_fake_docker_backend(monkeypatch)
    client = _build_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/review",
        json={
            "repo_path": str(repo_path),
            "intent": "fix",
            "sandbox_backend": "docker",
            "user_query": "Can you improve the README.md of this repo?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["fix_result"]["applied"] is True
    assert "README.md" in payload["fix_result"]["changed_files"]
    updated_readme = (repo_path / "README.md").read_text()
    assert "## Overview" in updated_readme
    assert "## Setup" in updated_readme
    assert "## Testing" in updated_readme
    assert "python app.py" in updated_readme
    assert "Describe the purpose" not in updated_readme
    assert "Document how" not in updated_readme


def test_fix_mode_refreshes_placeholder_readme_sections(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / ".env.example").write_text("APP_ENV=development\n")
    (repo_path / "tennis_petting_zoo.ipynb").write_text("{}\n")
    (repo_path / "README.md").write_text(
        "# Tennis PettingZoo Demo\n\n"
        "A notebook demo.\n\n"
        "## Installation\n\n"
        "Open the notebook.\n\n"
        "## Usage\n\n"
        "Run the cells sequentially.\n\n"
        "## Project Structure\n\n"
        "List the main folders, entrypoints, and important modules for new contributors.\n\n"
        "## Testing\n\n"
        "Document the commands contributors should run before shipping changes.\n"
    )

    _enable_fake_docker_backend(monkeypatch)
    client = _build_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/review",
        json={
            "repo_path": str(repo_path),
            "intent": "fix",
            "sandbox_backend": "docker",
            "user_query": "Can you edit and improve the README.md of this repo?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["fix_result"]["applied"] is True
    assert "README.md" in payload["fix_result"]["changed_files"]
    updated_readme = (repo_path / "README.md").read_text()
    assert "tennis_petting_zoo.ipynb" in updated_readme
    assert "running all cells from top to bottom" in updated_readme
    assert "List the main folders" not in updated_readme
    assert "Document the commands contributors should run" not in updated_readme


def test_fix_mode_uses_llm_fix_agent_for_readme_generation(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "README.md").write_text("# Demo\n\nShort description.\n")
    (repo_path / "app.py").write_text("print('demo')\n")

    class FakeFixSubAgent:
        def __init__(self, settings):
            del settings

        async def generate_fixes(self, **kwargs):
            del kwargs
            return FixAgentResult(
                summary="Generated a stronger README.",
                changes=[
                    FixChange(
                        path="README.md",
                        reason="Replace the short README with project-specific onboarding.",
                        fixed_finding_titles=[],
                        content=(
                            "# Demo\n\n"
                            "A small Python demo application.\n\n"
                            "## Overview\n\n"
                            "This project demonstrates a simple Python entry point.\n\n"
                            "## Setup\n\n"
                            "Create a virtual environment before running locally.\n\n"
                            "## Usage\n\n"
                            "```bash\npython app.py\n```\n\n"
                            "## Project Structure\n\n"
                            "- `app.py` - Main application entry point.\n\n"
                            "## Testing\n\n"
                            "Run a smoke check with `python app.py`.\n"
                        ),
                    )
                ],
                backend=ProviderBackend.NVIDIA,
                used_llm=True,
            )

    monkeypatch.setattr("agent_swarms.services.fix_mode.FixSubAgent", FakeFixSubAgent)
    _enable_fake_docker_backend(monkeypatch)
    client = _build_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/review",
        json={
            "repo_path": str(repo_path),
            "intent": "fix",
            "sandbox_backend": "docker",
            "user_query": "Can you improve the README.md of this repo?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["fix_result"]["applied"] is True
    assert "README.md" in payload["fix_result"]["changed_files"]
    updated_readme = (repo_path / "README.md").read_text()
    assert "A small Python demo application." in updated_readme
    assert "Replace the short README" in payload["fix_result"]["summary"]


def test_fix_mode_uses_llm_fix_agent_for_safe_code_edits(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / ".env.example").write_text("APP_ENV=development\n")
    (repo_path / "README.md").write_text("# Demo\n\nInstall\nSetup\nRun\nTest\n")
    (repo_path / "main.py").write_text("def calculate():\n    return eval('1 + 1')\n")

    class FakeFixSubAgent:
        def __init__(self, settings):
            del settings

        async def generate_fixes(self, **kwargs):
            del kwargs
            return FixAgentResult(
                summary="Generated a targeted code fix.",
                changes=[
                    FixChange(
                        path="main.py",
                        reason="Replace unsafe eval with direct arithmetic.",
                        fixed_finding_titles=[
                            "Potentially unsafe execution or deserialization patterns detected"
                        ],
                        content="def calculate():\n    return 1 + 1\n",
                    )
                ],
                backend=ProviderBackend.NVIDIA,
                used_llm=True,
            )

    monkeypatch.setattr("agent_swarms.services.fix_mode.FixSubAgent", FakeFixSubAgent)
    _enable_fake_docker_backend(monkeypatch)
    client = _build_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/review",
        json={
            "repo_path": str(repo_path),
            "intent": "fix",
            "sandbox_backend": "docker",
            "user_query": "Please fix unsafe code in this repo.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["fix_result"]["applied"] is True
    assert "main.py" in payload["fix_result"]["changed_files"]
    assert "eval" not in (repo_path / "main.py").read_text()
