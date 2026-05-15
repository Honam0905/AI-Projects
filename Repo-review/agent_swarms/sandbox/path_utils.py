"""Path helpers for sandboxed workspaces."""

from pathlib import Path


def resolve_within_root(root: Path, relative_path: str) -> Path:
    """Resolve a relative path while preventing escape outside the root."""

    candidate = (root / relative_path).resolve()
    resolved_root = root.resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ValueError(f"Path '{relative_path}' escapes the sandbox workspace.")
    return candidate


def ensure_existing_directory(path: Path) -> Path:
    """Validate that a path exists and is a directory."""

    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Expected a directory path, got: {path}")
    return path
