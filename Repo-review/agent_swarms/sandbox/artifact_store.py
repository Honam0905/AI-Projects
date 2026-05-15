"""Persistent artifact storage for sandbox runs."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_swarms.sandbox.base import SandboxSession
from agent_swarms.state.schemas import (
    SandboxArtifactEntry,
    SandboxArtifactsResponse,
    SandboxRunResponse,
)


class ArtifactStore:
    """Persist run metadata and execution artifacts to disk."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._runs_dir = data_dir / "runs"
        self._runs_dir.mkdir(parents=True, exist_ok=True)

    def build_workspace_path(self, run_id: str) -> Path:
        """Return the workspace path for a run."""

        return self._run_dir(run_id) / "workspace"

    def create_run_dirs(self, run_id: str) -> None:
        """Create the run directory and artifact directory."""

        run_dir = self._run_dir(run_id)
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    def write_run_metadata(self, session: SandboxSession) -> None:
        """Persist run metadata to disk."""

        payload = {
            "run_id": session.run_id,
            "sandbox_backend": session.sandbox_backend.value,
            "status": session.status.value,
            "workspace_path": str(session.workspace_path),
            "repo_source_path": str(session.repo_source_path) if session.repo_source_path else None,
            "sandbox_id": session.sandbox_id,
            "created_at": session.created_at.isoformat(),
            "closed_at": session.closed_at.isoformat() if session.closed_at else None,
        }
        self._write_json(self._run_dir(session.run_id) / "run.json", payload)

    def read_run_metadata(self, run_id: str) -> dict[str, Any]:
        """Load persisted run metadata."""

        path = self._run_dir(run_id) / "run.json"
        if not path.exists():
            raise FileNotFoundError(f"Sandbox run '{run_id}' was not found.")
        return json.loads(path.read_text())

    def write_execution_artifact(
        self,
        run_id: str,
        kind: str,
        payload: dict[str, Any],
    ) -> str:
        """Persist an execution payload and return its relative artifact path."""

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        filename = f"{timestamp}-{kind}.json"
        relative_path = Path("artifacts") / filename
        self._write_json(self._run_dir(run_id) / relative_path, payload)
        return str(relative_path)

    def write_text_artifact(
        self,
        run_id: str,
        kind: str,
        content: str,
        *,
        suffix: str = ".txt",
    ) -> str:
        """Persist a text artifact and return its relative path."""

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        filename = f"{timestamp}-{kind}{suffix}"
        relative_path = Path("artifacts") / filename
        path = self._run_dir(run_id) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return str(relative_path)

    def snapshot_workspace(self, run_id: str, snapshot_name: str) -> Path:
        """Copy the current workspace into a snapshot directory and return the path."""

        source = self.build_workspace_path(run_id)
        target = self._run_dir(run_id) / "snapshots" / snapshot_name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        return target

    def list_artifacts(self, run_id: str) -> SandboxArtifactsResponse:
        """List artifacts for a run."""

        artifacts_dir = self._run_dir(run_id) / "artifacts"
        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Sandbox run '{run_id}' was not found.")

        artifacts = [
            SandboxArtifactEntry(
                path=str(path.relative_to(self._run_dir(run_id))),
                size_bytes=path.stat().st_size,
                created_at=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
            )
            for path in sorted(artifacts_dir.iterdir())
            if path.is_file()
        ]
        return SandboxArtifactsResponse(run_id=run_id, artifacts=artifacts)

    def read_text_artifact(self, run_id: str, artifact_path: str) -> str:
        """Read one persisted text artifact."""

        run_dir = self._run_dir(run_id).resolve()
        path = (run_dir / artifact_path).resolve()
        if not path.is_relative_to(run_dir) or not path.is_file():
            raise FileNotFoundError(
                f"Artifact '{artifact_path}' was not found for review run '{run_id}'."
            )
        return path.read_text()

    def build_run_response(self, run_id: str) -> SandboxRunResponse:
        """Build an API response model from persisted metadata."""

        payload = self.read_run_metadata(run_id)
        return SandboxRunResponse.model_validate(payload)

    def _run_dir(self, run_id: str) -> Path:
        return self._runs_dir / run_id

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
