"""Safe automatic fix helpers for Docker-backed review runs."""

from __future__ import annotations

from dataclasses import dataclass
import difflib
from pathlib import Path
import re
import shlex

from agent_swarms.agents.fix_subagent import FixSubAgent
from agent_swarms.config import Settings
from agent_swarms.sandbox.service import SandboxService
from agent_swarms.services.report_builder import build_review_report
from agent_swarms.state.enums import WorkerRole
from agent_swarms.state.schemas import (
    FixResult,
    RepoMapSummary,
    ReviewFinding,
    ReviewReport,
    ReviewWorkerSummary,
    ShellExecutionRequest,
)

README_MISSING_TITLE = "README is missing"
README_SECTIONS_TITLE = "README is present but missing common onboarding sections"
ENV_EXAMPLE_TITLE = "Environment example file is missing"
PLACEHOLDER_README_PHRASES = (
    "Describe the purpose",
    "Document how",
    "List the required",
    "Explain how",
    "Document the primary",
    "Add the dependency installation",
    "Document the required configuration",
    "Document the commands contributors should run",
    "List the main folders",
    "Add concise notes",
)


@dataclass(frozen=True)
class FixExecution:
    """Normalized output of the automatic fix pass."""

    report: ReviewReport
    fix_result: FixResult


async def apply_fix_mode(
    *,
    run_id: str,
    sandbox_service: SandboxService,
    report: ReviewReport,
    user_query: str,
    settings: Settings | None = None,
) -> FixExecution:
    """Apply the supported automatic fixes and return the updated report."""

    baseline_path = sandbox_service.snapshot_workspace(run_id, "pre-fix")
    repo_map = report.repo_map.model_copy(deep=True)
    findings = [finding.model_copy(deep=True) for finding in report.top_findings]
    changed_files: list[str] = []
    fixed_titles: list[str] = []
    action_notes: list[str] = []
    action_notes_by_path: dict[str, str] = {}
    workspace_name = Path(sandbox_service.get_workspace_path(run_id)).name
    env_example = _find_finding(findings, ENV_EXAMPLE_TITLE)
    readme_context = repo_map.model_copy(deep=True)
    if env_example is not None:
        _append_unique(readme_context.top_level_files, "./.env.example")

    llm_execution = await FixSubAgent(settings or Settings()).generate_fixes(
        run_id=run_id,
        sandbox_service=sandbox_service,
        report=report,
        user_query=user_query,
        repo_name=workspace_name,
    )
    llm_changed_python_files: list[str] = []
    previous_contents: dict[str, str | None] = {}
    fixed_titles_by_path: dict[str, list[str]] = {}
    for change in llm_execution.changes:
        existing_content = await _safe_read_file(sandbox_service, run_id, change.path)
        if existing_content == change.content:
            continue
        previous_contents[change.path] = existing_content
        await sandbox_service.write_file(run_id, change.path, change.content)
        _append_unique(changed_files, change.path)
        _append_repo_map_file(repo_map, change.path)
        fixed_titles_by_path[change.path] = change.fixed_finding_titles
        fixed_titles.extend(
            title for title in change.fixed_finding_titles if title not in fixed_titles
        )
        action_note = f"LLM fix agent updated {change.path}: {change.reason}"
        action_notes.append(action_note)
        action_notes_by_path[change.path] = action_note
        if change.path.endswith(".py"):
            llm_changed_python_files.append(change.path)

    if llm_changed_python_files:
        failed_python_files = await _rollback_invalid_python_changes(
            run_id=run_id,
            sandbox_service=sandbox_service,
            changed_paths=llm_changed_python_files,
            previous_contents=previous_contents,
        )
        if failed_python_files:
            changed_files = [
                path for path in changed_files if path not in set(failed_python_files)
            ]
            failed_titles = {
                title
                for path in failed_python_files
                for title in fixed_titles_by_path.get(path, [])
            }
            fixed_titles = [title for title in fixed_titles if title not in failed_titles]
            failed_notes = {
                note
                for path in failed_python_files
                if (note := action_notes_by_path.get(path))
            }
            action_notes = [note for note in action_notes if note not in failed_notes]
            action_notes.append(
                "Skipped unsafe Python code edit because syntax validation failed."
            )

    readme_already_changed = _readme_changed(changed_files)
    readme_missing = _find_finding(findings, README_MISSING_TITLE)
    if readme_missing is not None and readme_already_changed:
        fixed_titles.append(README_MISSING_TITLE)
    elif readme_missing is not None:
        readme_path = "README.md"
        await sandbox_service.write_file(
            run_id,
            readme_path,
            _build_readme_content(repo_name=workspace_name, repo_map=readme_context),
        )
        _append_unique(repo_map.top_level_files, f"./{readme_path}")
        _append_unique(repo_map.docs_files, f"./{readme_path}")
        _append_unique(changed_files, readme_path)
        fixed_titles.append(README_MISSING_TITLE)
        action_notes.append("Created README.md with install, setup, run, and test sections.")

    readme_sections = _find_finding(findings, README_SECTIONS_TITLE)
    if readme_sections is not None and readme_already_changed:
        fixed_titles.append(README_SECTIONS_TITLE)
    elif readme_sections is not None:
        readme_path = _extract_repo_file_path(readme_sections) or "README.md"
        existing_readme = (await sandbox_service.read_file(run_id, readme_path)).content
        missing_sections = _extract_missing_sections(readme_sections.summary)
        updated_readme = _enhance_readme(
            existing=existing_readme,
            repo_name=workspace_name,
            repo_map=readme_context,
            preferred_sections=missing_sections,
        )
        if updated_readme != existing_readme:
            await sandbox_service.write_file(run_id, readme_path, updated_readme)
            _append_unique(repo_map.top_level_files, f"./{readme_path}")
            _append_unique(repo_map.docs_files, f"./{readme_path}")
            _append_unique(changed_files, readme_path)
            fixed_titles.append(README_SECTIONS_TITLE)
            sections_label = ", ".join(missing_sections) or "common onboarding"
            action_notes.append(
                f"Expanded {readme_path} with: {sections_label}."
            )

    if _query_targets_readme(user_query) and not readme_already_changed:
        readme_path = "README.md"
        existing_readme = ""
        try:
            existing_readme = (await sandbox_service.read_file(run_id, readme_path)).content
        except FileNotFoundError:
            existing_readme = ""
        updated_readme = _enhance_readme(
            existing=existing_readme,
            repo_name=workspace_name,
            repo_map=readme_context,
            preferred_sections=["overview", "setup", "usage", "project structure", "testing"],
            refresh_placeholders=True,
        )
        if updated_readme != existing_readme:
            await sandbox_service.write_file(run_id, readme_path, updated_readme)
            _append_unique(repo_map.top_level_files, f"./{readme_path}")
            _append_unique(repo_map.docs_files, f"./{readme_path}")
            _append_unique(changed_files, readme_path)
            action_notes.append(
                "Improved README.md with clearer overview, setup, usage, "
                "project structure, and testing guidance."
            )

    if env_example is not None:
        env_path = ".env.example"
        await sandbox_service.write_file(
            run_id,
            env_path,
            _build_env_example_content(),
        )
        _append_unique(repo_map.top_level_files, f"./{env_path}")
        _append_unique(changed_files, env_path)
        fixed_titles.append(ENV_EXAMPLE_TITLE)
        action_notes.append("Created .env.example with safe placeholder values.")

    if not changed_files:
        remaining_titles = [finding.title for finding in findings]
        return FixExecution(
            report=report,
            fix_result=FixResult(
                applied=False,
                summary=(
                    "Fix mode ran, but it did not find a safe supported edit for this request. "
                    "It currently handles README improvements and safe .env.example creation."
                ),
                changed_files=[],
                fixed_finding_titles=[],
                unsupported_finding_titles=remaining_titles,
                diff_artifact_path=None,
            ),
        )

    diff_text = build_workspace_diff(
        before_root=baseline_path,
        after_root=sandbox_service.get_workspace_path(run_id),
    )
    diff_artifact_path = sandbox_service.write_text_artifact(
        run_id,
        "fix-diff",
        diff_text or "No textual diff was generated.",
        suffix=".diff",
    )

    remaining_findings = [
        finding
        for finding in findings
        if finding.title not in set(fixed_titles)
    ]
    worker_summaries = [
        *report.worker_summaries,
        ReviewWorkerSummary(
            agent_name=WorkerRole.DOCS_DEVEX_REVIEWER,
            summary=(
                f"Applied {len(changed_files)} safe fix"
                f"{'' if len(changed_files) == 1 else 'es'} in the sandbox."
            ),
            commands_run=[],
            artifacts=[diff_artifact_path],
        ),
    ]
    updated_report = build_review_report(
        repo_map=repo_map,
        findings=remaining_findings,
        worker_summaries=worker_summaries,
        commands_run=report.commands_run,
        artifacts=report.artifacts,
    )
    fix_result = FixResult(
        applied=True,
        summary=(
            f"Applied {len(changed_files)} safe fix"
            f"{'' if len(changed_files) == 1 else 'es'}: {' '.join(action_notes)}"
        ),
        changed_files=sorted(set(changed_files)),
        fixed_finding_titles=fixed_titles,
        unsupported_finding_titles=[finding.title for finding in remaining_findings],
        diff_artifact_path=diff_artifact_path,
    )
    return FixExecution(report=updated_report, fix_result=fix_result)


async def _safe_read_file(
    sandbox_service: SandboxService,
    run_id: str,
    relative_path: str,
) -> str | None:
    try:
        return (await sandbox_service.read_file(run_id, relative_path)).content
    except FileNotFoundError:
        return None


def _append_repo_map_file(repo_map: RepoMapSummary, relative_path: str) -> None:
    normalized = f"./{relative_path.removeprefix('./')}"
    _append_unique(repo_map.top_level_files, normalized)
    if Path(relative_path).name.lower() in {"readme.md", "readme"}:
        _append_unique(repo_map.docs_files, normalized)


def _readme_changed(changed_files: list[str]) -> bool:
    return any(Path(path).name.lower() in {"readme.md", "readme"} for path in changed_files)


async def _rollback_invalid_python_changes(
    *,
    run_id: str,
    sandbox_service: SandboxService,
    changed_paths: list[str],
    previous_contents: dict[str, str | None],
) -> list[str]:
    command = "python3 -m py_compile " + " ".join(
        shlex.quote(path) for path in changed_paths
    )
    validation = await sandbox_service.execute_shell(
        run_id,
        ShellExecutionRequest(
            command=command,
            worker_role=WorkerRole.DOCS_DEVEX_REVIEWER,
            tool_name="llm_fix_validation_tool",
        ),
    )
    if validation.exit_code == 0:
        return []

    for relative_path in changed_paths:
        previous_content = previous_contents.get(relative_path)
        if previous_content is not None:
            await sandbox_service.write_file(run_id, relative_path, previous_content)
    return changed_paths


def build_workspace_diff(*, before_root: Path, after_root: Path) -> str:
    """Build a unified diff between two workspace snapshots."""

    relative_paths = sorted(
        {
            *(_iter_text_files(before_root)),
            *(_iter_text_files(after_root)),
        }
    )
    chunks: list[str] = []
    for relative_path in relative_paths:
        before_path = before_root / relative_path
        after_path = after_root / relative_path
        before_lines = _read_text_lines(before_path)
        after_lines = _read_text_lines(after_path)
        if before_lines == after_lines:
            continue
        chunks.extend(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=f"a/{relative_path.as_posix()}",
                tofile=f"b/{relative_path.as_posix()}",
                lineterm="",
            )
        )
        chunks.append("")
    return "\n".join(chunks).strip()


def _iter_text_files(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and _is_text_file(path)
    }


def _read_text_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text().splitlines()
    except UnicodeDecodeError:
        return []


def _is_text_file(path: Path) -> bool:
    try:
        path.read_text()
    except UnicodeDecodeError:
        return False
    return True


def _find_finding(findings: list[ReviewFinding], title: str) -> ReviewFinding | None:
    return next((finding for finding in findings if finding.title == title), None)


def _extract_repo_file_path(finding: ReviewFinding) -> str | None:
    if finding.file_paths:
        return finding.file_paths[0].removeprefix("./")
    if finding.evidence:
        first = finding.evidence[0].strip()
        if first.startswith("./"):
            return first.removeprefix("./")
    return None


def _extract_missing_sections(summary: str) -> list[str]:
    marker = "does not mention:"
    if marker not in summary.lower():
        return ["install", "setup", "run", "test"]
    _, _, tail = summary.partition(":")
    return [
        item.strip().strip(".")
        for item in tail.split(",")
        if item.strip()
    ]


def _append_missing_readme_sections(
    existing: str,
    sections: list[str],
    repo_map: RepoMapSummary,
    *,
    refresh_placeholders: bool = False,
) -> str:
    normalized = existing.rstrip()
    existing_lower = normalized.lower()
    additions: list[str] = []
    for section in sections:
        replacement = _build_readme_section(section, repo_map)
        if _readme_has_section(existing_lower, section):
            if refresh_placeholders:
                normalized = _replace_placeholder_section(
                    normalized,
                    section=section,
                    replacement=replacement,
                )
                existing_lower = normalized.lower()
            continue
        additions.append(replacement)
    if not additions:
        return normalized if normalized != existing.rstrip() else existing
    return f"{normalized}\n\n" + "\n\n".join(additions) + "\n"


def _readme_has_section(existing_lower: str, section: str) -> bool:
    return any(
        re.search(rf"^##\s+{alias}\s*$", existing_lower, flags=re.MULTILINE)
        for alias in _section_aliases(section)
    )


def _replace_placeholder_section(markdown: str, *, section: str, replacement: str) -> str:
    aliases = "|".join(_section_aliases(section))
    pattern = re.compile(
        rf"^##\s+(?:{aliases})\s*\n(?P<body>.*?)(?=^##\s+|\Z)",
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(markdown)
    if match is None:
        return markdown
    if not _section_looks_placeholder(match.group("body")):
        return markdown
    return (
        markdown[: match.start()].rstrip()
        + "\n\n"
        + replacement.rstrip()
        + "\n\n"
        + markdown[match.end():].lstrip()
    ).strip() + "\n"


def _section_looks_placeholder(body: str) -> bool:
    return any(phrase.lower() in body.lower() for phrase in PLACEHOLDER_README_PHRASES)


def _section_aliases(section: str) -> list[str]:
    normalized = section.strip().lower()
    aliases = {
        "install": ["install", "installation"],
        "setup": ["setup", "configuration"],
        "usage": ["usage", "run"],
        "run": ["run", "usage"],
        "test": ["test", "testing"],
        "testing": ["test", "testing"],
        "project structure": ["project\\s+structure"],
        "overview": ["overview"],
    }
    return aliases.get(normalized, [re.escape(normalized).replace("\\ ", r"\s+")])


def _enhance_readme(
    *,
    existing: str,
    repo_name: str,
    repo_map: RepoMapSummary,
    preferred_sections: list[str],
    refresh_placeholders: bool = False,
) -> str:
    if not existing.strip():
        return _build_readme_content(repo_name=repo_name, repo_map=repo_map)

    ordered_sections = list(preferred_sections)
    if "overview" not in {item.lower() for item in ordered_sections}:
        ordered_sections.insert(0, "overview")
    if "project structure" not in {item.lower() for item in ordered_sections}:
        ordered_sections.append("project structure")
    if "testing" not in {item.lower() for item in ordered_sections}:
        ordered_sections.append("testing")
    return _append_missing_readme_sections(
        existing,
        ordered_sections,
        repo_map,
        refresh_placeholders=refresh_placeholders,
    )


def _build_readme_content(*, repo_name: str, repo_map: RepoMapSummary) -> str:
    title = repo_name.replace("-", " ").replace("_", " ").strip().title() or "Project"
    sections = [
        f"# {title}",
        _build_overview_section(repo_name=title, repo_map=repo_map),
        _build_install_section(repo_map),
        _build_setup_section(repo_map),
        _build_usage_section(repo_map),
        _build_testing_section(repo_map),
        _build_project_structure_section(repo_map),
    ]
    return "\n\n".join(section.rstrip() for section in sections) + "\n"


def _build_readme_section(section: str, repo_map: RepoMapSummary | None = None) -> str:
    heading = section.strip().title()
    if repo_map is None:
        return f"## {heading}\n\nAdd concise notes for this workflow as the project evolves."

    body_map = {
        "Install": _build_install_section(repo_map),
        "Overview": _build_overview_section(repo_name="This project", repo_map=repo_map),
        "Setup": _build_setup_section(repo_map),
        "Usage": _build_usage_section(repo_map),
        "Run": _build_usage_section(repo_map, heading="Run"),
        "Testing": _build_testing_section(repo_map),
        "Test": _build_testing_section(repo_map, heading="Test"),
        "Project Structure": _build_project_structure_section(repo_map),
    }
    return body_map.get(
        heading,
        f"## {heading}\n\nAdd concise notes for this workflow as the project evolves.",
    )


def _build_overview_section(*, repo_name: str, repo_map: RepoMapSummary) -> str:
    language_label = _format_language_stack(repo_map.languages)
    sentences = [f"{repo_name} is a {language_label}."]
    if repo_map.entry_points:
        entry_points = _format_inline_list(repo_map.entry_points, 3)
        sentences.append(f"Main entry points detected: {entry_points}.")
    if repo_map.test_locations:
        sentences.append(f"Tests detected in {_format_inline_list(repo_map.test_locations, 3)}.")
    elif repo_map.mapped_file_count:
        sentences.append("No obvious test entry point was detected during the automated scan.")
    return "## Overview\n\n" + " ".join(sentences)


def _build_install_section(repo_map: RepoMapSummary) -> str:
    commands: list[str] = []
    paths = _all_repo_paths(repo_map)
    if "requirements.txt" in paths:
        commands.extend(
            [
                "python -m venv .venv",
                "source .venv/bin/activate",
                "pip install -r requirements.txt",
            ]
        )
    elif "pyproject.toml" in paths:
        commands.extend(
            [
                "python -m venv .venv",
                "source .venv/bin/activate",
                "pip install -e .",
            ]
        )
    if "package.json" in paths:
        commands.append("npm install")

    if commands:
        return (
            "## Install\n\nInstall dependencies from the repository root.\n\n"
            + _command_block(commands)
        )

    return (
        "## Install\n\n"
        "No standard dependency manifest was detected in this scan. "
        "Use the package manager expected by this project, then add the command here "
        "when it is known."
    )


def _build_setup_section(repo_map: RepoMapSummary) -> str:
    paths = _all_repo_paths(repo_map)
    if ".env.example" in paths:
        return (
            "## Setup\n\n"
            "Create a local environment file from the safe example before running the app.\n\n"
            + _command_block(["cp .env.example .env"])
            + "\n\nKeep real credentials in `.env` and do not commit them."
        )

    return (
        "## Setup\n\n"
        "No required environment template was detected. "
        "Keep real credentials in a local `.env` file and commit only safe placeholder values."
    )


def _build_usage_section(repo_map: RepoMapSummary, *, heading: str = "Usage") -> str:
    commands = _detect_run_commands(repo_map)
    if commands:
        return f"## {heading}\n\nStart from the repository root.\n\n" + _command_block(commands)

    if repo_map.entry_points:
        return (
            f"## {heading}\n\n"
            f"Start with the detected entry point: {_format_inline_list(repo_map.entry_points, 1)}."
        )

    return (
        f"## {heading}\n\n"
        "No default run command was detected in this scan. Add the main local run command here."
    )


def _build_testing_section(repo_map: RepoMapSummary, *, heading: str = "Testing") -> str:
    command = _detect_test_command(repo_map)
    if command:
        body = f"Run the test suite before opening a pull request.\n\n{_command_block([command])}"
    elif _notebook_paths(repo_map):
        notebook = _notebook_paths(repo_map)[0]
        body = (
            "No automated test suite was detected. For this notebook demo, validate the workflow "
            f"by opening `{notebook}` and running all cells from top to bottom."
        )
    elif repo_map.test_locations:
        body = (
            f"Tests were detected in {_format_inline_list(repo_map.test_locations, 3)}, "
            "but no standard test command was inferred."
        )
    else:
        body = "No test directory or test command was detected in this scan."
    return f"## {heading}\n\n{body}"


def _build_project_structure_section(repo_map: RepoMapSummary) -> str:
    items: list[str] = []
    if repo_map.entry_points:
        entry_point = repo_map.entry_points[0].removeprefix("./")
        items.append(f"- `{entry_point}` - Main detected entry point.")
    if repo_map.docs_files:
        items.append(f"- `{repo_map.docs_files[0].removeprefix('./')}` - Project documentation.")
    if repo_map.test_locations:
        test_path = repo_map.test_locations[0].removeprefix("./")
        items.append(f"- `{test_path}` - Test coverage detected by the scan.")
    if repo_map.dependency_files:
        dependency_file = repo_map.dependency_files[0].removeprefix("./")
        items.append(f"- `{dependency_file}` - Dependency manifest.")
    if not items:
        items.append("- `.` - Repository root.")
    return "## Project Structure\n\n" + "\n".join(items)


def _detect_run_commands(repo_map: RepoMapSummary) -> list[str]:
    paths = _all_repo_paths(repo_map)
    if "main.py" in paths:
        return ["python main.py"]
    if "app.py" in paths:
        return ["python app.py"]
    if "manage.py" in paths:
        return ["python manage.py runserver"]
    if _notebook_paths(repo_map):
        return [f"jupyter notebook {_notebook_paths(repo_map)[0]}"]
    if "package.json" in paths:
        return ["npm run dev"]
    return []


def _detect_test_command(repo_map: RepoMapSummary) -> str | None:
    paths = _all_repo_paths(repo_map)
    if "package.json" in paths and any("test" in path for path in paths):
        return "npm test"
    if repo_map.test_locations or {"requirements.txt", "pyproject.toml"} & paths:
        return "pytest"
    return None


def _notebook_paths(repo_map: RepoMapSummary) -> list[str]:
    return [
        path.removeprefix("./")
        for path in repo_map.entry_points + repo_map.top_level_files
        if path.lower().endswith(".ipynb")
    ]


def _all_repo_paths(repo_map: RepoMapSummary) -> set[str]:
    values = [
        *repo_map.top_level_files,
        *repo_map.dependency_files,
        *repo_map.entry_points,
        *repo_map.test_locations,
        *repo_map.docs_files,
    ]
    normalized = {value.removeprefix("./").lower() for value in values}
    normalized.update(Path(value).name.lower() for value in normalized)
    return normalized


def _format_language_stack(languages: list[str]) -> str:
    clean = [language.strip().lower() for language in languages if language.strip()]
    if not clean:
        return "software repository"
    if len(clean) == 1:
        return f"{clean[0]} project"
    return f"multi-language project using {', '.join(clean[:-1])}, and {clean[-1]}"


def _format_inline_list(values: list[str], limit: int) -> str:
    visible = [f"`{value.removeprefix('./')}`" for value in values[:limit]]
    if len(values) > limit:
        visible.append(f"+{len(values) - limit} more")
    return ", ".join(visible)


def _command_block(commands: list[str]) -> str:
    return "```bash\n" + "\n".join(commands) + "\n```"


def _build_env_example_content() -> str:
    return (
        "# Example environment configuration\n"
        "# Copy this file to .env and replace placeholder values before running locally.\n\n"
        "APP_ENV=development\n"
        "APP_PORT=8000\n"
        "LOG_LEVEL=INFO\n"
        "# DATABASE_URL=\n"
        "# API_KEY=\n"
    )


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _query_targets_readme(user_query: str) -> bool:
    normalized = user_query.lower()
    return "readme" in normalized and any(
        keyword in normalized
        for keyword in {"improve", "update", "rewrite", "fix", "edit", "enhance"}
    )
