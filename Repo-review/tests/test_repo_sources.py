"""Tests for repository source normalization and GitHub PR checkout support."""

import asyncio
from pathlib import Path
import subprocess

from agent_swarms.config import Settings
from agent_swarms.services.repo_sources import RepoSourceResolver
from agent_swarms.state.schemas import ReviewRequest


def test_repo_source_resolver_supports_github_pr_urls(monkeypatch, tmp_path: Path) -> None:
    repo_cache_dir = tmp_path / "repo-cache"
    settings = Settings(
        sandbox_data_dir=str(tmp_path / ".agent_swarms_data"),
        repo_cache_dir=str(repo_cache_dir),
    )
    resolver = RepoSourceResolver(settings)
    commands: list[tuple[str, ...]] = []

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text, timeout, cwd
        commands.append(tuple(args))
        if args[:3] == ["git", "clone", "--depth"]:
            Path(args[-1]).mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = resolver._clone_remote_repo("https://github.com/acme/demo/pull/42")

    assert result.repo_source == "https://github.com/acme/demo/pull/42"
    assert result.repo_path.startswith(str(repo_cache_dir))
    assert commands[0][:4] == ("git", "clone", "--depth", "1")
    assert commands[1] == (
        "git",
        "fetch",
        "--depth",
        "1",
        "origin",
        "pull/42/head:refs/heads/pr-42",
    )
    assert commands[2] == ("git", "checkout", "pr-42")


def test_repo_source_resolver_rejects_multi_project_parent_folder(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    first_repo = root / "travel-planner"
    second_repo = root / "crypto-trade-agent"
    first_repo.mkdir(parents=True)
    second_repo.mkdir(parents=True)
    (first_repo / "pyproject.toml").write_text("[project]\nname='travel-planner'\n")
    (second_repo / "package.json").write_text('{"name":"crypto-trade-agent"}\n')

    resolver = RepoSourceResolver(Settings(sandbox_data_dir=str(tmp_path / ".agent_swarms_data")))

    request = ReviewRequest(repo_path=str(root), user_query="Review this repository.")
    try:
        asyncio.run(resolver.resolve(request))
    except ValueError as exc:
        assert "multiple repositories" in str(exc)
    else:
        raise AssertionError("Expected the resolver to reject a parent workspace path.")


def test_repo_source_resolver_rejects_parent_folder_even_with_root_marker(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    first_repo = root / "travel-planner"
    second_repo = root / "crypto-trade-agent"
    first_repo.mkdir(parents=True)
    second_repo.mkdir(parents=True)
    (root / "requirements.txt").write_text("pytest\n")
    (first_repo / "pyproject.toml").write_text("[project]\nname='travel-planner'\n")
    (second_repo / "package.json").write_text('{"name":"crypto-trade-agent"}\n')

    resolver = RepoSourceResolver(Settings(sandbox_data_dir=str(tmp_path / ".agent_swarms_data")))
    request = ReviewRequest(repo_path=str(root), user_query="Review this repository.")

    try:
        asyncio.run(resolver.resolve(request))
    except ValueError as exc:
        assert "Choose one exact repo root" in str(exc)
    else:
        raise AssertionError("Expected loose parent folders with child repo roots to be rejected.")


def test_repo_source_resolver_uses_local_clone_for_remote_fix(tmp_path: Path) -> None:
    search_root = tmp_path / "clones"
    clone_path = search_root / "turboquant-pytorch"
    git_dir = clone_path / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/tonbistudio/turboquant-pytorch.git\n'
    )

    settings = Settings(
        sandbox_data_dir=str(tmp_path / ".agent_swarms_data"),
        local_clone_search_roots=str(search_root),
    )
    resolver = RepoSourceResolver(settings)
    request = ReviewRequest(
        repo_url="https://github.com/tonbistudio/turboquant-pytorch",
        user_query="Fix the README.",
        intent="fix",
    )

    result = asyncio.run(resolver.resolve(request))

    assert result.repo_source == "https://github.com/tonbistudio/turboquant-pytorch"
    assert result.repo_path == str(clone_path.resolve())
    assert result.repo_source_type.value == "remote"


def test_repo_source_resolver_requires_local_clone_for_remote_fix(tmp_path: Path) -> None:
    settings = Settings(
        sandbox_data_dir=str(tmp_path / ".agent_swarms_data"),
        local_clone_search_roots=str(tmp_path / "empty"),
    )
    resolver = RepoSourceResolver(settings)
    request = ReviewRequest(
        repo_url="https://github.com/tonbistudio/turboquant-pytorch",
        user_query="Fix the README.",
        intent="fix",
    )

    try:
        asyncio.run(resolver.resolve(request))
    except ValueError as exc:
        assert str(exc) == "Remote fix mode requires a matching local clone."
    else:
        raise AssertionError("Expected remote fix mode to require a local clone.")
