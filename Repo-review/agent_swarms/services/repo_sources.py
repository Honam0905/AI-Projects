"""Repository source resolution for local and remote review targets."""

from __future__ import annotations

import asyncio
import configparser
import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from agent_swarms.config import Settings
from agent_swarms.sandbox.base import SandboxConfigurationError
from agent_swarms.sandbox.path_utils import ensure_existing_directory
from agent_swarms.state.enums import RepoSourceType, ReviewIntent
from agent_swarms.state.schemas import ReviewRequest

GITHUB_HTTP_PATTERN = re.compile(
    r"^https://(?:www\.)?github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)(?:\.git)?/?$",
    re.IGNORECASE,
)
GITHUB_PR_PATTERN = re.compile(
    r"^https://(?:www\.)?github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)/pull/(?P<number>\d+)/?$",
    re.IGNORECASE,
)
GITHUB_SSH_PATTERN = re.compile(
    r"^git@github\.com:(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)(?:\.git)?$",
    re.IGNORECASE,
)
README_NAMES = {"readme.md", "readme"}


@dataclass(frozen=True)
class ResolvedRepoSource:
    """A normalized repository source that points to a local directory."""

    repo_path: str
    repo_source_type: RepoSourceType
    repo_source: str


@dataclass(frozen=True)
class RemoteCloneTarget:
    """Normalized remote target for repo or PR cloning."""

    clone_url: str
    display_url: str
    pr_number: int | None = None


class RepoSourceResolver:
    """Resolve a review request into a local path the sandbox can stage."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def resolve(self, request: ReviewRequest) -> ResolvedRepoSource:
        """Resolve either a local directory or a remote clone."""

        if request.repo_url:
            if request.intent is ReviewIntent.FIX:
                return await asyncio.to_thread(self._resolve_remote_fix_repo, request.repo_url)
            return await asyncio.to_thread(self._clone_remote_repo, request.repo_url)
        return self._resolve_local_repo(request.repo_path)

    def _resolve_local_repo(self, repo_path: str | None) -> ResolvedRepoSource:
        source = repo_path or self._settings.default_review_repo_path
        resolved_path = ensure_existing_directory(Path(source).expanduser().resolve())
        self._validate_repo_root(resolved_path)
        resolved_value = str(resolved_path)
        return ResolvedRepoSource(
            repo_path=resolved_value,
            repo_source_type=RepoSourceType.LOCAL,
            repo_source=resolved_value,
        )

    def _resolve_remote_fix_repo(self, repo_url: str) -> ResolvedRepoSource:
        target = self._normalize_remote_url(repo_url)
        local_clone = self._find_local_clone(target)
        if local_clone is None:
            raise ValueError("Remote fix mode requires a matching local clone.")
        return ResolvedRepoSource(
            repo_path=str(local_clone),
            repo_source_type=RepoSourceType.REMOTE,
            repo_source=target.display_url,
        )

    def _clone_remote_repo(self, repo_url: str) -> ResolvedRepoSource:
        target = self._normalize_remote_url(repo_url)
        repo_dir = self._cache_root() / self._cache_key(target.display_url)

        if repo_dir.exists():
            shutil.rmtree(repo_dir)

        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", target.clone_url, str(repo_dir)],
                check=True,
                capture_output=True,
                text=True,
                timeout=self._settings.repo_clone_timeout_seconds,
            )
            if target.pr_number is not None:
                self._checkout_pull_request(repo_dir=repo_dir, target=target)
        except FileNotFoundError as exc:
            raise SandboxConfigurationError(
                "Git is required to clone remote repositories."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ValueError(
                f"Timed out while cloning remote repository: {target.display_url}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            detail = stderr.splitlines()[-1] if stderr else "Unknown git clone failure."
            raise ValueError(
                f"Failed to clone remote repository '{target.display_url}': {detail}"
            ) from exc

        resolved_path = ensure_existing_directory(repo_dir.resolve())
        return ResolvedRepoSource(
            repo_path=str(resolved_path),
            repo_source_type=RepoSourceType.REMOTE,
            repo_source=target.display_url,
        )

    def _normalize_remote_url(self, repo_url: str) -> RemoteCloneTarget:
        value = repo_url.strip()
        parsed = urlparse(value)

        if parsed.scheme == "file":
            file_path = ensure_existing_directory(Path(unquote(parsed.path)).expanduser().resolve())
            file_url = file_path.as_uri()
            return RemoteCloneTarget(clone_url=file_url, display_url=file_url)

        pr_match = GITHUB_PR_PATTERN.match(value)
        if pr_match:
            owner = pr_match.group("owner")
            repo = pr_match.group("repo")
            number = int(pr_match.group("number"))
            return RemoteCloneTarget(
                clone_url=f"https://github.com/{owner}/{repo}.git",
                display_url=f"https://github.com/{owner}/{repo}/pull/{number}",
                pr_number=number,
            )

        http_match = GITHUB_HTTP_PATTERN.match(value)
        if http_match:
            owner = http_match.group("owner")
            repo = http_match.group("repo")
            return RemoteCloneTarget(
                clone_url=f"https://github.com/{owner}/{repo}.git",
                display_url=f"https://github.com/{owner}/{repo}",
            )

        ssh_match = GITHUB_SSH_PATTERN.match(value)
        if ssh_match:
            owner = ssh_match.group("owner")
            repo = ssh_match.group("repo")
            return RemoteCloneTarget(
                clone_url=f"https://github.com/{owner}/{repo}.git",
                display_url=f"https://github.com/{owner}/{repo}",
            )

        raise ValueError(
            "Unsupported repo_url. Use a GitHub repository URL such as "
            "https://github.com/owner/repo or a pull request URL such as "
            "https://github.com/owner/repo/pull/123."
        )

    def _checkout_pull_request(self, *, repo_dir: Path, target: RemoteCloneTarget) -> None:
        if target.pr_number is None:
            return

        subprocess.run(
            [
                "git",
                "fetch",
                "--depth",
                "1",
                "origin",
                f"pull/{target.pr_number}/head:refs/heads/pr-{target.pr_number}",
            ],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=self._settings.repo_clone_timeout_seconds,
        )
        subprocess.run(
            ["git", "checkout", f"pr-{target.pr_number}"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=self._settings.repo_clone_timeout_seconds,
        )

    def _cache_root(self) -> Path:
        if self._settings.repo_cache_dir:
            root = Path(self._settings.repo_cache_dir).expanduser().resolve()
        else:
            root = Path(self._settings.sandbox_data_dir).expanduser().resolve() / "repo-cache"
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _cache_key(value: str) -> str:
        parsed = urlparse(value)
        stem = Path(parsed.path.rstrip("/")).name or "repo"
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-") or "repo"
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
        return f"{safe_stem}-{digest}"

    def _validate_repo_root(self, repo_path: Path) -> None:
        nested_repo_roots = self._nested_repo_roots(repo_path)
        if (
            len(nested_repo_roots) >= 2
            and not self._looks_like_monorepo_root(repo_path)
        ):
            self._raise_parent_folder_error(repo_path, nested_repo_roots)

        if self._looks_like_repo_root(repo_path):
            return

        if len(nested_repo_roots) < 2:
            return

        self._raise_parent_folder_error(repo_path, nested_repo_roots)

    def _nested_repo_roots(self, repo_path: Path) -> list[Path]:
        return [
            child
            for child in sorted(repo_path.iterdir())
            if child.is_dir()
            and not child.name.startswith(".")
            and self._looks_like_nested_project_root(child)
        ]

    def _find_local_clone(self, target: RemoteCloneTarget) -> Path | None:
        repo_name = Path(urlparse(target.display_url).path.rstrip("/")).name
        expected_identity = self._normalize_repo_identity(target.display_url)
        for root in self._local_clone_search_roots():
            for candidate in self._iter_named_directories(root, repo_name):
                if not self._looks_like_git_checkout(candidate):
                    continue
                remote_url = self._read_origin_url(candidate)
                if remote_url is None:
                    continue
                if self._normalize_repo_identity(remote_url) != expected_identity:
                    continue
                return candidate.resolve()
        return None

    def _local_clone_search_roots(self) -> list[Path]:
        configured = self._settings.local_clone_search_roots
        values = (
            [item.strip() for item in re.split(r"[,;]", configured) if item.strip()]
            if configured
            else [
                str(Path.cwd().resolve().parent),
                str(Path(self._settings.default_review_repo_path).expanduser().resolve().parent),
            ]
        )
        roots: list[Path] = []
        seen: set[Path] = set()
        for value in values:
            path = Path(value).expanduser().resolve()
            if not path.is_dir() or path in seen:
                continue
            seen.add(path)
            roots.append(path)
        return roots

    @staticmethod
    def _iter_named_directories(root: Path, directory_name: str, *, max_depth: int = 5):
        for current_root, dir_names, _ in os.walk(root):
            current_path = Path(current_root)
            relative = current_path.relative_to(root)
            depth = len(relative.parts)
            dir_names[:] = [
                name
                for name in dir_names
                if name not in {".git", "node_modules", ".venv", "__pycache__"}
            ]
            if depth > max_depth:
                dir_names[:] = []
                continue
            if current_path.name == directory_name:
                yield current_path

    @staticmethod
    def _looks_like_git_checkout(path: Path) -> bool:
        git_path = path / ".git"
        return git_path.is_dir() or git_path.is_file()

    @staticmethod
    def _looks_like_nested_project_root(path: Path) -> bool:
        return any(
            (path / marker).exists()
            for marker in {
                ".git",
                "pyproject.toml",
                "requirements.txt",
                "package.json",
                "go.mod",
                "Cargo.toml",
                "Gemfile",
                "composer.json",
            }
        )

    @staticmethod
    def _looks_like_repo_root(path: Path) -> bool:
        marker_names = {
            ".git",
            "pyproject.toml",
            "requirements.txt",
            "package.json",
            "pnpm-workspace.yaml",
            "turbo.json",
            "nx.json",
            "go.mod",
            "Cargo.toml",
            "Gemfile",
            "composer.json",
        }
        source_dir_names = {"src", "app", "apps", "lib", "tests", "docs"}
        top_level_files = [child for child in path.iterdir() if child.is_file()]
        top_level_dirs = [child for child in path.iterdir() if child.is_dir()]
        if any((path / marker).exists() for marker in marker_names):
            return True
        source_extensions = {".py", ".js", ".ts", ".tsx", ".go", ".rs", ".java"}
        if any(file_path.suffix.lower() in source_extensions for file_path in top_level_files):
            return True
        has_readme = any(file_path.name.lower() in README_NAMES for file_path in top_level_files)
        has_source_dir = any(
            child.name.lower() in source_dir_names for child in top_level_dirs
        )
        return has_readme and has_source_dir

    @staticmethod
    def _looks_like_monorepo_root(path: Path) -> bool:
        return any(
            (path / marker).exists()
            for marker in {
                "pnpm-workspace.yaml",
                "turbo.json",
                "nx.json",
                "lerna.json",
            }
        )

    @staticmethod
    def _raise_parent_folder_error(repo_path: Path, nested_repo_roots: list[Path]) -> None:
        sample_names = ", ".join(child.name for child in nested_repo_roots[:4])
        if len(nested_repo_roots) > 4:
            sample_names = f"{sample_names}, +{len(nested_repo_roots) - 4} more"
        raise ValueError(
            f"'{repo_path}' looks like a parent folder that contains multiple repositories "
            f"or app roots ({sample_names}). Choose one exact repo root instead."
        )

    def _read_origin_url(self, repo_path: Path) -> str | None:
        git_path = repo_path / ".git"
        config_path = git_path / "config" if git_path.is_dir() else None
        if git_path.is_file():
            pointer = git_path.read_text().strip()
            if pointer.startswith("gitdir:"):
                git_dir = (repo_path / pointer.partition(":")[2].strip()).resolve()
                config_path = git_dir / "config"
        if config_path is None or not config_path.exists():
            return None

        parser = configparser.ConfigParser()
        parser.read(config_path)
        for section in parser.sections():
            if section.lower() == 'remote "origin"' and parser.has_option(section, "url"):
                return parser.get(section, "url")
        return None

    def _normalize_repo_identity(self, repo_url: str) -> str:
        target = self._normalize_remote_url(repo_url)
        return target.display_url.removesuffix("/")
