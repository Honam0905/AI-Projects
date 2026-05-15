"""Workspace preparation for local sandbox runs."""

from __future__ import annotations

import shutil
from pathlib import Path

from agent_swarms.sandbox.path_utils import ensure_existing_directory

IGNORE_NAMES = (
    ".agent_swarms_data",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".uv-cache",
    ".venv",
    "node_modules",
    ".next",
    "dist",
    "build",
    "coverage",
)


class WorkspaceManager:
    """Manage per-run workspace directories."""

    def prepare_workspace(self, workspace_path: Path, repo_source_path: Path | None) -> None:
        """Create a clean workspace and optionally copy a local repo into it."""

        workspace_path.mkdir(parents=True, exist_ok=True)
        if repo_source_path is None:
            return

        source_dir = ensure_existing_directory(repo_source_path)
        shutil.copytree(
            source_dir,
            workspace_path,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*IGNORE_NAMES),
        )
