"""Unit tests for sandbox adapter edge cases."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_swarms.config import Settings
from agent_swarms.sandbox.base import SandboxExecutionError, SandboxSession
from agent_swarms.sandbox.docker_adapter import DockerSandboxAdapter
from agent_swarms.sandbox.local_adapter import LocalSandboxAdapter
from agent_swarms.sandbox.opensandbox_adapter import OpenSandboxAdapter
from agent_swarms.sandbox.workspace_manager import WorkspaceManager
from agent_swarms.state.enums import SandboxBackend, SandboxRunStatus


class _SlowCommands:
    async def run(self, command: str) -> SimpleNamespace:
        del command
        await asyncio.sleep(0.05)
        return SimpleNamespace(
            id="cmd-1",
            logs=SimpleNamespace(stdout=[], stderr=[]),
        )

    async def get_command_status(self, execution_id: str) -> SimpleNamespace:
        del execution_id
        return SimpleNamespace(exit_code=0, started_at=None, finished_at=None)


def test_opensandbox_execute_shell_honors_timeout() -> None:
    adapter = object.__new__(OpenSandboxAdapter)
    session = SandboxSession(
        run_id="run-1",
        sandbox_backend=SandboxBackend.OPENSANDBOX,
        workspace_path=Path("/workspace"),
        created_at=datetime.now(UTC),
        status=SandboxRunStatus.READY,
        handle=SimpleNamespace(commands=_SlowCommands()),
    )

    with pytest.raises(SandboxExecutionError, match="timed out"):
        asyncio.run(adapter.execute_shell(session, "sleep 10", timeout_seconds=0.01))


def test_docker_adapter_requires_docker_cli(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None if name == "docker" else None)
    adapter = DockerSandboxAdapter(
        Settings(sandbox_data_dir=str(tmp_path / ".agent_swarms_data")),
        WorkspaceManager(),
    )

    with pytest.raises(Exception, match="Docker CLI is required"):
        asyncio.run(adapter.create_session("run-1"))


def test_docker_adapter_mounts_workspace_and_assigns_container(monkeypatch, tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "README.md").write_text("# Demo\n")
    workspace_path = tmp_path / "workspace"

    monkeypatch.setattr(
        "shutil.which",
        lambda name: "/usr/bin/docker" if name == "docker" else None,
    )
    adapter = DockerSandboxAdapter(
        Settings(
            sandbox_data_dir=str(tmp_path / ".agent_swarms_data"),
            docker_sandbox_container_prefix="tests",
        ),
        WorkspaceManager(),
    )
    monkeypatch.setattr(adapter, "_ensure_docker_available", lambda: None)
    monkeypatch.setattr(adapter, "_ensure_runtime_image", lambda: None)

    removed: list[str] = []
    started: list[str] = []
    monkeypatch.setattr(adapter, "_remove_container", lambda name: removed.append(name))
    monkeypatch.setattr(
        adapter,
        "_start_container",
        lambda session: started.append(session.sandbox_id or ""),
    )

    session = SandboxSession(
        run_id="run-1234567890abcdef",
        sandbox_backend=SandboxBackend.DOCKER,
        workspace_path=workspace_path,
        created_at=datetime.now(UTC),
        status=SandboxRunStatus.READY,
    )

    mounted_session = asyncio.run(adapter.mount_repo(session, repo_path))

    assert (workspace_path / "README.md").exists()
    assert mounted_session.sandbox_id == "tests-run-12345678"
    assert removed == ["tests-run-12345678"]
    assert started == ["tests-run-12345678"]


def test_local_adapter_wraps_macos_profile_with_key_value_definition(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "shutil.which",
        lambda name: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None,
    )
    adapter = LocalSandboxAdapter(WorkspaceManager(), use_macos_profile=True)

    wrapped = adapter._wrap_with_macos_profile(
        args=("/bin/sh", "-lc", "pwd"),
        cwd=tmp_path,
    )

    assert wrapped[0] == "/usr/bin/sandbox-exec"
    assert wrapped[1:4] == ("-D", f"WORKSPACE={tmp_path.resolve()}", "-p")
    assert wrapped[-3:] == ("/bin/sh", "-lc", "pwd")


def test_local_adapter_falls_back_for_execvp_operation_not_permitted() -> None:
    adapter = LocalSandboxAdapter(WorkspaceManager(), use_macos_profile=True)

    assert adapter._should_fallback_from_profile(
        b"sandbox-exec: execvp() of '/bin/sh' failed: Operation not permitted",
        1,
    )
