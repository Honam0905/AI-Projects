"""OpenSandbox adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from datetime import UTC, datetime
from pathlib import Path

from agent_swarms.config import Settings
from agent_swarms.sandbox.base import (
    ExecutionRecord,
    SandboxAdapter,
    SandboxConfigurationError,
    SandboxExecutionError,
    SandboxFileRecord,
    SandboxSession,
)
from agent_swarms.state.enums import SandboxBackend, SandboxRunStatus


class OpenSandboxAdapter(SandboxAdapter):
    """Adapter for OpenSandbox-backed execution."""

    backend = SandboxBackend.OPENSANDBOX

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sdk = self._load_sdk()

    async def create_session(self, run_id: str) -> SandboxSession:
        """Create a remote OpenSandbox session."""

        sandbox = await self._sdk["Sandbox"].create(
            self._settings.open_sandbox_template,
            connection_config=self._sdk["ConnectionConfig"](
                domain=self._settings.open_sandbox_domain,
                api_key=self._settings.open_sandbox_api_key or "",
                protocol=self._settings.open_sandbox_protocol,
            ),
        )

        return SandboxSession(
            run_id=run_id,
            sandbox_backend=self.backend,
            workspace_path=Path("/workspace"),
            created_at=datetime.now(UTC),
            status=SandboxRunStatus.READY,
            sandbox_id=sandbox.id,
            handle=sandbox,
        )

    async def mount_repo(
        self,
        session: SandboxSession,
        repo_source_path: Path | None,
    ) -> SandboxSession:
        """Upload local repository files into the remote workspace."""

        if repo_source_path is None:
            return session

        source_root = repo_source_path.resolve()
        if not source_root.exists() or not source_root.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {repo_source_path}")

        write_entry = self._sdk["WriteEntry"]
        entries = []
        for source_file in source_root.rglob("*"):
            if source_file.is_dir():
                continue
            try:
                data = source_file.read_text()
            except UnicodeDecodeError:
                continue

            relative_path = source_file.relative_to(source_root)
            remote_path = session.workspace_path / relative_path
            entries.append(
                write_entry(
                    path=str(remote_path),
                    data=data,
                    mode=0o644,
                )
            )

        if entries:
            await session.handle.files.write_files(entries)

        session.repo_source_path = repo_source_path
        return session

    async def execute_shell(
        self,
        session: SandboxSession,
        command: str,
        timeout_seconds: int,
    ) -> ExecutionRecord:
        """Execute a shell command in the remote workspace."""

        return await self._run_with_timeout(
            self._execute_remote_command(
                session=session,
                command=command,
                execution_command=f"cd {session.workspace_path} && {command}",
            ),
            timeout_seconds=timeout_seconds,
            command=command,
        )

    async def execute_python(
        self,
        session: SandboxSession,
        code: str,
        timeout_seconds: int,
        python_executable: str,
    ) -> ExecutionRecord:
        """Execute Python code in the remote workspace."""

        command = (
            f"cd {session.workspace_path} && "
            f"{python_executable} - <<'PY'\n{code}\nPY"
        )
        return await self._run_with_timeout(
            self._execute_remote_command(
                session=session,
                command=command,
                execution_command=command,
            ),
            timeout_seconds=timeout_seconds,
            command=command,
        )

    async def list_files(
        self,
        session: SandboxSession,
        relative_path: str,
    ) -> list[SandboxFileRecord]:
        """List files in the remote workspace."""

        search_entry = self._sdk["SearchEntry"]
        target_path = session.workspace_path / relative_path
        results = await session.handle.files.search(
            search_entry(path=str(target_path), pattern="*")
        )
        return [
            SandboxFileRecord(
                path=str(Path(item.path).relative_to(session.workspace_path)),
                is_dir=False,
                size_bytes=getattr(item, "size", 0),
            )
            for item in results
        ]

    async def read_file(
        self,
        session: SandboxSession,
        relative_path: str,
    ) -> str:
        """Read a text file from the remote workspace."""

        target_path = session.workspace_path / relative_path
        return await session.handle.files.read_file(str(target_path))

    async def write_file(
        self,
        session: SandboxSession,
        relative_path: str,
        content: str,
    ) -> None:
        """Write a text file into the remote workspace."""

        write_entry = self._sdk["WriteEntry"]
        target_path = session.workspace_path / relative_path
        await session.handle.files.write_files(
            [
                write_entry(
                    path=str(target_path),
                    data=content,
                    mode=0o644,
                )
            ]
        )

    async def close_session(self, session: SandboxSession) -> SandboxSession:
        """Terminate the remote sandbox session."""

        if session.handle is not None:
            await session.handle.kill()
        session.status = SandboxRunStatus.CLOSED
        session.closed_at = datetime.now(UTC)
        return session

    def _load_sdk(self) -> dict[str, object]:
        try:
            from opensandbox.config import ConnectionConfig
            from opensandbox.models.filesystem import SearchEntry, WriteEntry
            from opensandbox.sandbox import Sandbox
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise SandboxConfigurationError(
                "The opensandbox package is required for sandbox_backend=opensandbox."
            ) from exc

        return {
            "ConnectionConfig": ConnectionConfig,
            "Sandbox": Sandbox,
            "SearchEntry": SearchEntry,
            "WriteEntry": WriteEntry,
        }

    async def _execute_remote_command(
        self,
        *,
        session: SandboxSession,
        command: str,
        execution_command: str,
    ) -> ExecutionRecord:
        try:
            execution = await session.handle.commands.run(execution_command)
        except Exception as exc:  # pragma: no cover - external dependency
            raise SandboxExecutionError(str(exc)) from exc

        return await self._normalize_execution(session, command, execution)

    async def _normalize_execution(
        self,
        session: SandboxSession,
        command: str,
        execution: object,
    ) -> ExecutionRecord:
        stdout = self._join_logs(getattr(getattr(execution, "logs", None), "stdout", []))
        stderr = self._join_logs(getattr(getattr(execution, "logs", None), "stderr", []))
        execution_id = getattr(execution, "id", None)
        exit_code = 0
        duration_ms = 0

        if execution_id:
            status = await session.handle.commands.get_command_status(execution_id)
            exit_code = int(status.exit_code or 0)
            if status.started_at and status.finished_at:
                duration_ms = int((status.finished_at - status.started_at).total_seconds() * 1000)

        return ExecutionRecord(
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            executed_at=datetime.now(UTC),
        )

    def _join_logs(self, logs: list[object]) -> str:
        return "".join(getattr(item, "text", "") for item in logs)

    async def _run_with_timeout(
        self,
        operation: Awaitable[ExecutionRecord],
        *,
        timeout_seconds: int,
        command: str,
    ) -> ExecutionRecord:
        try:
            return await asyncio.wait_for(operation, timeout=timeout_seconds)
        except TimeoutError as exc:
            raise SandboxExecutionError(
                f"Command timed out after {timeout_seconds} seconds: {command}"
            ) from exc
