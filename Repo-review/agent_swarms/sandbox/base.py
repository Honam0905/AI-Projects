"""Base contracts for sandbox backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_swarms.state.enums import SandboxBackend, SandboxRunStatus


class SandboxConfigurationError(RuntimeError):
    """Raised when a sandbox backend is not configured correctly."""


class SandboxExecutionError(RuntimeError):
    """Raised when sandbox lifecycle or execution fails."""


@dataclass
class SandboxSession:
    """Active sandbox session metadata."""

    run_id: str
    sandbox_backend: SandboxBackend
    workspace_path: Path
    created_at: datetime
    status: SandboxRunStatus
    repo_source_path: Path | None = None
    sandbox_id: str | None = None
    closed_at: datetime | None = None
    handle: Any = None


@dataclass(frozen=True)
class ExecutionRecord:
    """Normalized execution result returned by sandbox backends."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    executed_at: datetime


@dataclass(frozen=True)
class SandboxFileRecord:
    """A file or directory inside the sandbox workspace."""

    path: str
    is_dir: bool
    size_bytes: int


class SandboxAdapter(ABC):
    """Abstract adapter for sandbox implementations."""

    backend: SandboxBackend

    @abstractmethod
    async def create_session(self, run_id: str) -> SandboxSession:
        """Create a new sandbox session."""

    @abstractmethod
    async def mount_repo(
        self,
        session: SandboxSession,
        repo_source_path: Path | None,
    ) -> SandboxSession:
        """Prepare the repo workspace inside the sandbox."""

    @abstractmethod
    async def execute_shell(
        self,
        session: SandboxSession,
        command: str,
        timeout_seconds: int,
    ) -> ExecutionRecord:
        """Run a shell command inside the sandbox."""

    @abstractmethod
    async def execute_python(
        self,
        session: SandboxSession,
        code: str,
        timeout_seconds: int,
        python_executable: str,
    ) -> ExecutionRecord:
        """Run Python code inside the sandbox."""

    @abstractmethod
    async def list_files(
        self,
        session: SandboxSession,
        relative_path: str,
    ) -> list[SandboxFileRecord]:
        """List files inside the sandbox workspace."""

    @abstractmethod
    async def read_file(
        self,
        session: SandboxSession,
        relative_path: str,
    ) -> str:
        """Read a file inside the sandbox workspace."""

    @abstractmethod
    async def write_file(
        self,
        session: SandboxSession,
        relative_path: str,
        content: str,
    ) -> None:
        """Write a file inside the sandbox workspace."""

    @abstractmethod
    async def close_session(self, session: SandboxSession) -> SandboxSession:
        """Close the sandbox session."""
