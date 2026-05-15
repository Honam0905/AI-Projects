"""Worker-level tests for repository mapping and docs/devex behavior."""

import asyncio
from types import SimpleNamespace

import pytest

from agent_swarms.agents.review_workers import run_repo_mapper, run_specialist
from agent_swarms.sandbox.base import SandboxExecutionError
from agent_swarms.state.enums import WorkerRole
from agent_swarms.state.schemas import RepoMapSummary


class _FakeSandboxService:
    def __init__(self, *, shell_response, files: dict[str, str] | None = None) -> None:
        if isinstance(shell_response, list):
            self._shell_responses = shell_response
        else:
            self._shell_responses = [shell_response]
        self._files = files or {}

    async def execute_shell(self, run_id: str, request):
        del run_id, request
        if len(self._shell_responses) == 1:
            return self._shell_responses[0]
        return self._shell_responses.pop(0)

    async def read_file(self, run_id: str, relative_path: str):
        del run_id
        return SimpleNamespace(content=self._files[relative_path])


def test_repo_mapper_raises_when_scan_command_fails() -> None:
    sandbox_service = _FakeSandboxService(
        shell_response=SimpleNamespace(
            command="find . -type f",
            exit_code=1,
            stdout="",
            stderr="find: permission denied",
            artifact_path="artifacts/mapper.json",
        )
    )

    with pytest.raises(SandboxExecutionError, match="Repository mapping failed"):
        asyncio.run(run_repo_mapper("run-1", sandbox_service))


def test_repo_mapper_raises_when_scan_returns_zero_files() -> None:
    sandbox_service = _FakeSandboxService(
        shell_response=SimpleNamespace(
            command="find . -type f",
            exit_code=0,
            stdout="",
            stderr="",
            artifact_path="artifacts/mapper.json",
        )
    )

    with pytest.raises(SandboxExecutionError, match="returned zero files"):
        asyncio.run(run_repo_mapper("run-1", sandbox_service))


def test_repo_mapper_captures_file_count_and_scan_stderr() -> None:
    sandbox_service = _FakeSandboxService(
        shell_response=SimpleNamespace(
            command="find . -type f",
            exit_code=0,
            stdout="./README.md\n./src/app.py\n",
            stderr="warning: skipped nothing",
            artifact_path="artifacts/mapper.json",
        )
    )

    result = asyncio.run(run_repo_mapper("run-1", sandbox_service))

    assert result.repo_map is not None
    assert result.repo_map.mapped_file_count == 2
    assert result.repo_map.languages == ["python"]
    assert result.repo_map.docs_files == ["./README.md"]
    assert result.repo_map.scan_stderr == "warning: skipped nothing"


def test_docs_devex_reviewer_skips_generic_missing_findings_for_empty_repo_map() -> None:
    repo_map = RepoMapSummary()
    sandbox_service = _FakeSandboxService(
        shell_response=SimpleNamespace(
            command="noop",
            exit_code=0,
            stdout="",
            stderr="",
            artifact_path="artifacts/noop.json",
        )
    )

    result = asyncio.run(
        run_specialist(
            worker_role=WorkerRole.DOCS_DEVEX_REVIEWER,
            run_id="run-1",
            sandbox_service=sandbox_service,
            repo_map=repo_map,
        )
    )

    assert result.findings == []
    assert "skipped" in result.summary.summary.lower()


def test_security_reviewer_ignores_config_variable_names_without_secret_values() -> None:
    sandbox_service = _FakeSandboxService(
        shell_response=[
            SimpleNamespace(
                command="secret scan",
                exit_code=0,
                stdout=(
                    "./agent_swarms/services/llm_provider.py:115:"
                    "nvidia_api_key=settings.nvidia_api_key\n"
                    "./agent_swarms/sandbox/opensandbox_adapter.py:36:"
                    "api_key=self._settings.open_sandbox_api_key or \"\"\n"
                ),
                stderr="",
                artifact_path="artifacts/secrets.json",
            ),
            SimpleNamespace(
                command="risky scan",
                exit_code=0,
                stdout="",
                stderr="",
                artifact_path="artifacts/risky.json",
            ),
        ]
    )

    result = asyncio.run(
        run_specialist(
            worker_role=WorkerRole.SECURITY_REVIEWER,
            run_id="run-1",
            sandbox_service=sandbox_service,
            repo_map=RepoMapSummary(mapped_file_count=2),
        )
    )

    assert result.findings == []


def test_security_reviewer_redacts_real_secret_values() -> None:
    sandbox_service = _FakeSandboxService(
        shell_response=[
            SimpleNamespace(
                command="secret scan",
                exit_code=0,
                stdout=(
                    "./src/config.py:1:"
                    "API_KEY=\"local_test_secret_value_1234567890\"\n"
                ),
                stderr="",
                artifact_path="artifacts/secrets.json",
            ),
            SimpleNamespace(
                command="risky scan",
                exit_code=0,
                stdout="",
                stderr="",
                artifact_path="artifacts/risky.json",
            ),
        ]
    )

    result = asyncio.run(
        run_specialist(
            worker_role=WorkerRole.SECURITY_REVIEWER,
            run_id="run-1",
            sandbox_service=sandbox_service,
            repo_map=RepoMapSummary(mapped_file_count=1),
        )
    )

    assert len(result.findings) == 1
    assert result.findings[0].file_paths == ["./src/config.py"]
    assert "<redacted-secret>" in result.findings[0].evidence[0]
