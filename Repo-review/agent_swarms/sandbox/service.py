"""High-level sandbox run service."""

from __future__ import annotations

import shutil
import threading
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from agent_swarms.config import Settings
from agent_swarms.observability.logging import get_logger
from agent_swarms.sandbox.artifact_store import ArtifactStore
from agent_swarms.sandbox.base import (
    SandboxAdapter,
    SandboxConfigurationError,
    SandboxSession,
)
from agent_swarms.sandbox.docker_adapter import DockerSandboxAdapter
from agent_swarms.sandbox.local_adapter import LocalSandboxAdapter
from agent_swarms.sandbox.opensandbox_adapter import OpenSandboxAdapter
from agent_swarms.sandbox.workspace_manager import WorkspaceManager
from agent_swarms.services.review_run_tracking import ReviewEventBroker
from agent_swarms.state.enums import RunEventType, RunStatus, SandboxBackend
from agent_swarms.state.schemas import (
    PythonExecutionRequest,
    SandboxExecutionResponse,
    SandboxFileContentResponse,
    SandboxFileEntry,
    SandboxFileListResponse,
    SandboxRunCreateRequest,
    SandboxRunResponse,
    ShellExecutionRequest,
)

logger = get_logger(__name__)


class SandboxService:
    """Manage sandbox-backed runs and artifact persistence."""

    def __init__(self, settings: Settings, event_broker: ReviewEventBroker | None = None) -> None:
        self._settings = settings
        self._artifact_store = ArtifactStore(Path(settings.sandbox_data_dir))
        self._workspace_manager = WorkspaceManager()
        self._active_runs: dict[str, SandboxSession] = {}
        self._event_broker = event_broker
        self._active_runs_lock = threading.Lock()

    async def create_run(self, request: SandboxRunCreateRequest) -> SandboxRunResponse:
        """Create a sandbox session and stage the requested repository."""

        run_id = uuid4().hex
        backend = request.sandbox_backend or self._settings.sandbox_backend
        try:
            self._artifact_store.create_run_dirs(run_id)
            workspace_path = self._artifact_store.build_workspace_path(run_id)
        except PermissionError as exc:
            raise SandboxConfigurationError(
                f"Sandbox data directory is not writable: {self._settings.sandbox_data_dir}"
            ) from exc

        repo_source_path = (
            Path(request.repo_path).expanduser().resolve()
            if request.repo_path is not None
            else None
        )

        adapter = self._build_adapter(backend)
        session = await adapter.create_session(run_id)
        if backend in {SandboxBackend.LOCAL, SandboxBackend.DOCKER}:
            session.workspace_path = workspace_path
        try:
            session = await adapter.mount_repo(session, repo_source_path)
        except (OSError, shutil.Error) as exc:
            raise SandboxConfigurationError(
                f"Failed to stage repository into the sandbox workspace: {exc}"
            ) from exc
        self._artifact_store.write_run_metadata(session)
        with self._active_runs_lock:
            self._active_runs[run_id] = session

        logger.info("Created sandbox run %s with backend=%s", run_id, backend.value)
        return self._artifact_store.build_run_response(run_id)

    def get_run(self, run_id: str) -> SandboxRunResponse:
        """Return persisted metadata for one sandbox run."""

        return self._artifact_store.build_run_response(run_id)

    async def execute_shell(
        self,
        run_id: str,
        request: ShellExecutionRequest,
    ) -> SandboxExecutionResponse:
        """Run a shell command through the selected sandbox adapter."""

        session = self._get_active_session(run_id)
        adapter = self._build_adapter(session.sandbox_backend)
        timeout_seconds = request.timeout_seconds or self._settings.sandbox_exec_timeout_seconds
        result = await adapter.execute_shell(session, request.command, timeout_seconds)
        artifact_path = self._artifact_store.write_execution_artifact(
            run_id,
            "shell",
            {
                "run_id": run_id,
                "sandbox_backend": session.sandbox_backend.value,
                "request": request.model_dump(),
                "result": asdict(result),
            },
        )
        await self._publish_tool_event(
            run_id=run_id,
            kind="shell",
            result=result,
            artifact_path=artifact_path,
            worker_role=request.worker_role.value if request.worker_role is not None else None,
            tool_name=request.tool_name,
        )
        return SandboxExecutionResponse(
            run_id=run_id,
            sandbox_backend=session.sandbox_backend,
            command=result.command,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_ms=result.duration_ms,
            artifact_path=artifact_path,
            executed_at=result.executed_at,
        )

    async def execute_python(
        self,
        run_id: str,
        request: PythonExecutionRequest,
    ) -> SandboxExecutionResponse:
        """Run Python code through the selected sandbox adapter."""

        session = self._get_active_session(run_id)
        adapter = self._build_adapter(session.sandbox_backend)
        timeout_seconds = request.timeout_seconds or self._settings.sandbox_exec_timeout_seconds
        python_executable = request.python_executable or self._settings.sandbox_python_executable
        result = await adapter.execute_python(
            session,
            request.code,
            timeout_seconds,
            python_executable,
        )
        artifact_path = self._artifact_store.write_execution_artifact(
            run_id,
            "python",
            {
                "run_id": run_id,
                "sandbox_backend": session.sandbox_backend.value,
                "request": request.model_dump(),
                "result": asdict(result),
            },
        )
        await self._publish_tool_event(
            run_id=run_id,
            kind="python",
            result=result,
            artifact_path=artifact_path,
            worker_role=request.worker_role.value if request.worker_role is not None else None,
            tool_name=request.tool_name,
        )
        return SandboxExecutionResponse(
            run_id=run_id,
            sandbox_backend=session.sandbox_backend,
            command=result.command,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_ms=result.duration_ms,
            artifact_path=artifact_path,
            executed_at=result.executed_at,
        )

    async def list_files(self, run_id: str, relative_path: str) -> SandboxFileListResponse:
        """List files below a relative path in the sandbox workspace."""

        session = self._get_active_session(run_id)
        adapter = self._build_adapter(session.sandbox_backend)
        entries = await adapter.list_files(session, relative_path)
        return SandboxFileListResponse(
            run_id=run_id,
            path=relative_path,
            entries=[
                SandboxFileEntry(
                    path=entry.path,
                    is_dir=entry.is_dir,
                    size_bytes=entry.size_bytes,
                )
                for entry in entries
            ],
        )

    async def read_file(self, run_id: str, relative_path: str) -> SandboxFileContentResponse:
        """Read a UTF-8 text file from the sandbox workspace."""

        session = self._get_active_session(run_id)
        adapter = self._build_adapter(session.sandbox_backend)
        content = await adapter.read_file(session, relative_path)
        return SandboxFileContentResponse(
            run_id=run_id,
            path=relative_path,
            content=content,
        )

    def list_artifacts(self, run_id: str) -> SandboxArtifactsResponse:
        """Return persisted artifacts for a sandbox run."""

        return self._artifact_store.list_artifacts(run_id)

    def read_artifact_text(self, run_id: str, artifact_path: str) -> str:
        """Read a persisted text artifact for a sandbox run."""

        return self._artifact_store.read_text_artifact(run_id, artifact_path)

    def write_artifact(self, run_id: str, kind: str, payload: dict[str, object]) -> str:
        """Persist a custom artifact for a sandbox run."""

        return self._artifact_store.write_execution_artifact(run_id, kind, payload)

    def write_text_artifact(
        self,
        run_id: str,
        kind: str,
        content: str,
        *,
        suffix: str = ".txt",
    ) -> str:
        """Persist a text artifact for a sandbox run."""

        return self._artifact_store.write_text_artifact(run_id, kind, content, suffix=suffix)

    def snapshot_workspace(self, run_id: str, snapshot_name: str) -> Path:
        """Capture a full copy of the current workspace for later diffing."""

        self._get_active_session(run_id)
        return self._artifact_store.snapshot_workspace(run_id, snapshot_name)

    def get_workspace_path(self, run_id: str) -> Path:
        """Return the live workspace path for an active run."""

        return self._get_active_session(run_id).workspace_path

    async def write_file(self, run_id: str, relative_path: str, content: str) -> None:
        """Write a text file inside the active sandbox workspace."""

        session = self._get_active_session(run_id)
        adapter = self._build_adapter(session.sandbox_backend)
        await adapter.write_file(session, relative_path, content)

    async def close_run(self, run_id: str) -> SandboxRunResponse:
        """Close the active sandbox session and persist final metadata."""

        session = self._get_active_session(run_id)
        adapter = self._build_adapter(session.sandbox_backend)
        session = await adapter.close_session(session)
        self._artifact_store.write_run_metadata(session)
        with self._active_runs_lock:
            self._active_runs.pop(run_id, None)
        return self._artifact_store.build_run_response(run_id)

    def _build_adapter(self, backend: SandboxBackend) -> SandboxAdapter:
        if backend is SandboxBackend.OPENSANDBOX:
            return OpenSandboxAdapter(self._settings)
        if backend is SandboxBackend.DOCKER:
            return DockerSandboxAdapter(self._settings, self._workspace_manager)
        return LocalSandboxAdapter(
            self._workspace_manager,
            use_macos_profile=self._settings.local_sandbox_use_macos_profile,
        )

    def _get_active_session(self, run_id: str) -> SandboxSession:
        with self._active_runs_lock:
            session = self._active_runs.get(run_id)
        if session is None:
            raise FileNotFoundError(
                f"Sandbox run '{run_id}' is not active in memory. Create a new "
                "run before executing tools."
            )
        return session

    async def _publish_tool_event(
        self,
        *,
        run_id: str,
        kind: str,
        result,
        artifact_path: str,
        worker_role: str | None = None,
        tool_name: str | None = None,
    ) -> None:
        if self._event_broker is None:
            return

        try:
            await self._event_broker.publish(
                run_id=run_id,
                event_type=RunEventType.TOOL_COMPLETED,
                message=f"{kind.title()} tool execution finished.",
                status=RunStatus.IN_PROGRESS,
                payload={
                    "kind": kind,
                    "command": result.command,
                    "exit_code": result.exit_code,
                    "duration_ms": result.duration_ms,
                    "artifact_path": artifact_path,
                    "worker_role": worker_role,
                    "tool_name": tool_name,
                },
            )
        except FileNotFoundError:
            logger.debug("Skipping tool event for untracked run %s", run_id)
