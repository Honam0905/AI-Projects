"""Docker-backed sandbox adapter."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from agent_swarms.config import Settings
from agent_swarms.sandbox.base import (
    ExecutionRecord,
    SandboxAdapter,
    SandboxConfigurationError,
    SandboxExecutionError,
    SandboxFileRecord,
    SandboxSession,
)
from agent_swarms.sandbox.path_utils import resolve_within_root
from agent_swarms.sandbox.workspace_manager import WorkspaceManager
from agent_swarms.state.enums import SandboxBackend, SandboxRunStatus


class DockerSandboxAdapter(SandboxAdapter):
    """Execute commands inside a long-lived Docker container."""

    backend = SandboxBackend.DOCKER

    def __init__(self, settings: Settings, workspace_manager: WorkspaceManager) -> None:
        self._settings = settings
        self._workspace_manager = workspace_manager
        self._docker_executable = shutil.which("docker")
        self._dockerfile_path = Path(__file__).with_name("docker_sandbox.Dockerfile")

    async def create_session(self, run_id: str) -> SandboxSession:
        """Create Docker-backed session metadata."""

        self._ensure_docker_available()
        return SandboxSession(
            run_id=run_id,
            sandbox_backend=self.backend,
            workspace_path=Path(),
            created_at=datetime.now(UTC),
            status=SandboxRunStatus.READY,
        )

    async def mount_repo(
        self,
        session: SandboxSession,
        repo_source_path: Path | None,
    ) -> SandboxSession:
        """Stage the repo and start a container for the session."""

        self._workspace_manager.prepare_workspace(session.workspace_path, repo_source_path)
        self._ensure_runtime_image()
        session.repo_source_path = repo_source_path
        session.sandbox_id = self._build_container_name(session.run_id)
        self._remove_container(session.sandbox_id)
        self._start_container(session)
        return session

    async def execute_shell(
        self,
        session: SandboxSession,
        command: str,
        timeout_seconds: int,
    ) -> ExecutionRecord:
        """Execute a shell command inside the running container."""

        return await self._exec_in_container(
            session=session,
            recorded_command=command,
            args=("/bin/sh", "-lc", command),
            timeout_seconds=timeout_seconds,
        )

    async def execute_python(
        self,
        session: SandboxSession,
        code: str,
        timeout_seconds: int,
        python_executable: str,
    ) -> ExecutionRecord:
        """Execute isolated Python code inside the running container."""

        script_path = session.workspace_path / f".sandbox-python-{uuid4().hex}.py"
        script_path.write_text(code)
        container_script_path = Path("/workspace") / script_path.name
        command = f"{python_executable} -I {container_script_path}"

        try:
            return await self._exec_in_container(
                session=session,
                recorded_command=command,
                args=(python_executable, "-I", str(container_script_path)),
                timeout_seconds=timeout_seconds,
            )
        finally:
            if script_path.exists():
                script_path.unlink()

    async def list_files(
        self,
        session: SandboxSession,
        relative_path: str,
    ) -> list[SandboxFileRecord]:
        """List files from the mounted workspace."""

        target_dir = resolve_within_root(session.workspace_path, relative_path)
        if not target_dir.exists():
            raise FileNotFoundError(f"Path does not exist: {relative_path}")
        if not target_dir.is_dir():
            raise ValueError(f"Path is not a directory: {relative_path}")

        return [
            SandboxFileRecord(
                path=str(path.relative_to(session.workspace_path)),
                is_dir=path.is_dir(),
                size_bytes=path.stat().st_size if path.is_file() else 0,
            )
            for path in sorted(target_dir.iterdir(), key=lambda item: item.name)
        ]

    async def read_file(
        self,
        session: SandboxSession,
        relative_path: str,
    ) -> str:
        """Read a text file from the mounted workspace."""

        target_file = resolve_within_root(session.workspace_path, relative_path)
        if not target_file.exists():
            raise FileNotFoundError(f"File does not exist: {relative_path}")
        if not target_file.is_file():
            raise ValueError(f"Path is not a file: {relative_path}")
        return target_file.read_text()

    async def write_file(
        self,
        session: SandboxSession,
        relative_path: str,
        content: str,
    ) -> None:
        """Write a text file into the mounted workspace."""

        target_file = resolve_within_root(session.workspace_path, relative_path)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(content)

    async def close_session(self, session: SandboxSession) -> SandboxSession:
        """Stop the container and mark the session closed."""

        if session.sandbox_id:
            self._remove_container(session.sandbox_id)
        session.status = SandboxRunStatus.CLOSED
        session.closed_at = datetime.now(UTC)
        return session

    async def _exec_in_container(
        self,
        *,
        session: SandboxSession,
        recorded_command: str,
        args: tuple[str, ...],
        timeout_seconds: int,
    ) -> ExecutionRecord:
        container_name = self._require_container_name(session)
        started_at = datetime.now(UTC)
        process = await asyncio.create_subprocess_exec(
            self._require_docker_executable(),
            "exec",
            container_name,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_raw, stderr_raw = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            self._restart_container(session)
            raise SandboxExecutionError(
                f"Command timed out after {timeout_seconds} seconds: {recorded_command}"
            ) from exc

        finished_at = datetime.now(UTC)
        return ExecutionRecord(
            command=recorded_command,
            exit_code=process.returncode,
            stdout=stdout_raw.decode("utf-8", errors="replace"),
            stderr=stderr_raw.decode("utf-8", errors="replace"),
            duration_ms=int((finished_at - started_at).total_seconds() * 1000),
            executed_at=started_at,
        )

    def _start_container(self, session: SandboxSession) -> None:
        container_name = self._require_container_name(session)
        self._run_docker_cli(
            [
                self._require_docker_executable(),
                "run",
                "-d",
                "--rm",
                "--name",
                container_name,
                "--network",
                "none",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--volume",
                f"{session.workspace_path.resolve()}:/workspace",
                "--workdir",
                "/workspace",
                self._settings.docker_sandbox_image,
            ],
            command_label="start Docker sandbox container",
        )

    def _restart_container(self, session: SandboxSession) -> None:
        container_name = self._require_container_name(session)
        self._remove_container(container_name)
        self._start_container(session)

    def _remove_container(self, container_name: str) -> None:
        self._run_docker_cli(
            [
                self._require_docker_executable(),
                "rm",
                "-f",
                container_name,
            ],
            command_label="remove Docker sandbox container",
            check=False,
        )

    def _ensure_docker_available(self) -> None:
        docker_executable = self._require_docker_executable()
        self._run_docker_cli(
            [docker_executable, "info"],
            command_label="check Docker availability",
            timeout_seconds=20,
        )

    def _ensure_runtime_image(self) -> None:
        docker_executable = self._require_docker_executable()
        inspect = self._run_docker_cli(
            [docker_executable, "image", "inspect", self._settings.docker_sandbox_image],
            command_label="inspect Docker sandbox image",
            check=False,
        )
        if inspect.returncode == 0:
            return

        self._run_docker_cli(
            [
                docker_executable,
                "build",
                "-t",
                self._settings.docker_sandbox_image,
                "-f",
                str(self._dockerfile_path),
                str(self._dockerfile_path.parent),
            ],
            command_label="build Docker sandbox image",
            timeout_seconds=900,
        )

    def _run_docker_cli(
        self,
        args: list[str],
        *,
        command_label: str,
        timeout_seconds: int = 60,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                args,
                check=check,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise SandboxConfigurationError(
                "Docker CLI is required for sandbox_backend=docker."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise SandboxConfigurationError(
                f"Timed out while trying to {command_label}."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            detail = stderr.splitlines()[-1] if stderr else "Unknown Docker error."
            raise SandboxConfigurationError(
                f"Failed to {command_label}: {detail}"
            ) from exc

    def _require_docker_executable(self) -> str:
        if self._docker_executable is None:
            raise SandboxConfigurationError("Docker CLI is required for sandbox_backend=docker.")
        return self._docker_executable

    def _require_container_name(self, session: SandboxSession) -> str:
        if not session.sandbox_id:
            raise SandboxExecutionError("Docker sandbox session is missing a container id.")
        return session.sandbox_id

    def _build_container_name(self, run_id: str) -> str:
        prefix = self._settings.docker_sandbox_container_prefix.strip() or "agent-swarms"
        return f"{prefix}-{run_id[:12]}"
