"""Deterministic sandbox tools used by LLM-assisted review sub-agents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from agent_swarms.agents.review_tool_registry import tool_for_worker
from agent_swarms.sandbox.base import SandboxExecutionError
from agent_swarms.sandbox.service import SandboxService
from agent_swarms.state.enums import FindingSeverity, WorkerRole
from agent_swarms.state.schemas import (
    RepoMapSummary,
    ReviewFinding,
    ReviewWorkerSummary,
    ShellExecutionRequest,
)

CODE_EXTENSIONS = {
    ".py": "python",
    ".ipynb": "notebook",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
}
DEPENDENCY_FILE_NAMES = {
    "pyproject.toml",
    "requirements.txt",
    "package.json",
    "poetry.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
}
ENTRYPOINT_NAMES = {"main.py", "app.py", "server.py", "manage.py", "package.json"}
README_NAMES = {"readme.md", "readme"}
FIND_EXCLUDE_ARGS = (
    "-not -path '*/.git/*' "
    "-not -path '*/node_modules/*' "
    "-not -path '*/.venv/*' "
    "-not -path '*/__pycache__/*' "
    "-not -path '*/.pytest_cache/*' "
    "-not -path '*/.next/*' "
    "-not -path '*/dist/*' "
    "-not -path '*/build/*' "
    "-not -path '*/coverage/*' "
)
RG_EXCLUDE_ARGS = (
    "--glob '!**/.git/**' "
    "--glob '!**/node_modules/**' "
    "--glob '!**/.venv/**' "
    "--glob '!**/__pycache__/**' "
    "--glob '!**/.pytest_cache/**' "
    "--glob '!**/.next/**' "
    "--glob '!**/dist/**' "
    "--glob '!**/build/**' "
    "--glob '!**/coverage/**' "
    "--glob '!**/.agent_swarms_data/**'"
)
SECRET_VALUE_PATTERNS = (
    re.compile(r"BEGIN [A-Z ]*PRIVATE KEY", re.IGNORECASE),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"nvapi-[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(
        r"\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*['\"]?"
        r"(?P<value>[A-Za-z0-9_./+=-]{20,})",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class WorkerExecution:
    """Normalized output returned by a worker."""

    findings: list[ReviewFinding]
    summary: ReviewWorkerSummary
    repo_map: RepoMapSummary | None = None


def _shell_request(
    worker_role: WorkerRole,
    command: str,
    *,
    timeout_seconds: int | None = None,
) -> ShellExecutionRequest:
    return ShellExecutionRequest(
        command=command,
        timeout_seconds=timeout_seconds,
        worker_role=worker_role,
        tool_name=_tool_name_for_role(worker_role),
    )


def _tool_name_for_role(worker_role: WorkerRole) -> str:
    return tool_for_worker(worker_role).name


async def run_repo_mapper(run_id: str, sandbox_service: SandboxService) -> WorkerExecution:
    """Build a lightweight repository map."""

    file_command = (
        f"find . -type f {FIND_EXCLUDE_ARGS}"
        "| sort | sed -n '1,400p'"
    )
    response = await sandbox_service.execute_shell(
        run_id,
        _shell_request(WorkerRole.REPO_MAPPER, file_command),
    )
    if response.exit_code != 0:
        detail = _trim_scan_stderr(response.stderr) or _trim_scan_stderr(response.stdout)
        suffix = f" Details: {detail}" if detail else ""
        raise SandboxExecutionError(
            "Repository mapping failed before the review workers could inspect "
            "the staged workspace."
            f"{suffix}"
        )

    files = [line.strip() for line in response.stdout.splitlines() if line.strip()]
    if not files:
        detail = _trim_scan_stderr(response.stderr)
        suffix = f" Details: {detail}" if detail else ""
        raise SandboxExecutionError(
            "Repository mapping returned zero files. The staged workspace may be "
            "empty, inaccessible, "
            f"or the scan command did not run as expected.{suffix}"
        )

    languages = sorted(
        {
            language
            for file_path in files
            for language in _detect_languages(file_path)
        }
    )
    dependency_files = [
        path
        for path in files
        if PurePosixPath(path).name in DEPENDENCY_FILE_NAMES
    ]
    entry_points = [
        path
        for path in files
        if PurePosixPath(path).name in ENTRYPOINT_NAMES
        or PurePosixPath(path).suffix.lower() == ".ipynb"
    ]
    test_locations = [path for path in files if "test" in path.lower()]
    docs_files = [
        path
        for path in files
        if PurePosixPath(path).name.lower() in README_NAMES
        or "/docs/" in path.lower()
    ]

    repo_map = RepoMapSummary(
        mapped_file_count=len(files),
        top_level_files=files[:40],
        languages=languages,
        dependency_files=dependency_files,
        entry_points=entry_points,
        test_locations=test_locations,
        docs_files=docs_files,
        scan_stderr=_trim_scan_stderr(response.stderr),
    )
    summary = ReviewWorkerSummary(
        agent_name=WorkerRole.REPO_MAPPER,
        summary=(
            f"Mapped {len(files)} files across the repository snapshot and detected "
            f"{', '.join(languages) or 'no clear language markers'}."
        ),
        commands_run=[response.command],
        artifacts=[response.artifact_path],
    )
    return WorkerExecution(findings=[], summary=summary, repo_map=repo_map)


async def run_specialist(
    *,
    worker_role: WorkerRole,
    run_id: str,
    sandbox_service: SandboxService,
    repo_map: RepoMapSummary,
) -> WorkerExecution:
    """Dispatch to the appropriate specialist worker."""

    if worker_role is WorkerRole.STATIC_REVIEWER:
        return await _run_static_reviewer(run_id, sandbox_service)
    if worker_role is WorkerRole.RUNTIME_TESTER:
        return await _run_runtime_tester(run_id, sandbox_service, repo_map)
    if worker_role is WorkerRole.SECURITY_REVIEWER:
        return await _run_security_reviewer(run_id, sandbox_service)
    if worker_role is WorkerRole.DOCS_DEVEX_REVIEWER:
        return await _run_docs_devex_reviewer(run_id, sandbox_service, repo_map)
    if worker_role is WorkerRole.EXTERNAL_VALIDATOR:
        return await _run_external_validator(run_id, sandbox_service, repo_map)
    raise ValueError(f"Unsupported worker role: {worker_role}")


async def _run_static_reviewer(run_id: str, sandbox_service: SandboxService) -> WorkerExecution:
    findings: list[ReviewFinding] = []
    artifacts: list[str] = []
    commands: list[str] = []

    large_file_response = await sandbox_service.execute_shell(
        run_id,
        _shell_request(
            WorkerRole.STATIC_REVIEWER,
            command=(
                "find . -type f \\( -name '*.py' -o -name '*.js' -o -name '*.ts' -o -name '*.tsx' "
                "-o -name '*.jsx' -o -name '*.go' -o -name '*.rs' -o -name '*.java' \\) "
                "-exec wc -l {} + 2>/dev/null | sort -nr | sed -n '1,20p'"
            ),
        ),
    )
    commands.append(large_file_response.command)
    artifacts.append(large_file_response.artifact_path)

    for line_count, file_path in _parse_wc_output(large_file_response.stdout):
        if line_count < 450:
            continue
        severity = FindingSeverity.MEDIUM if line_count >= 650 else FindingSeverity.LOW
        findings.append(
            ReviewFinding(
                title="Large source file may be hard to maintain",
                summary=(
                    f"{file_path} is {line_count} lines long, "
                    "which suggests a candidate for decomposition."
                ),
                severity=severity,
                confidence=0.72,
                source_agents=[WorkerRole.STATIC_REVIEWER],
                evidence=[f"{line_count} {file_path}"],
                file_paths=[file_path],
                suggested_fix=(
                    "Split large modules by responsibility so review and testing "
                    "stay manageable."
                ),
            )
        )

    todo_response = await sandbox_service.execute_shell(
        run_id,
        _shell_request(
            WorkerRole.STATIC_REVIEWER,
            'rg -n "TODO|FIXME|XXX|HACK" . || true',
        ),
    )
    commands.append(todo_response.command)
    artifacts.append(todo_response.artifact_path)

    todo_hits = [line for line in todo_response.stdout.splitlines() if line.strip()]
    if len(todo_hits) >= 3:
        findings.append(
            ReviewFinding(
                title="Open TODO or FIXME markers remain in the codebase",
                summary=(
                    f"Found {len(todo_hits)} TODO/FIXME-style markers that may "
                    "represent unfinished work."
                ),
                severity=FindingSeverity.LOW,
                confidence=0.65,
                source_agents=[WorkerRole.STATIC_REVIEWER],
                evidence=todo_hits[:10],
                suggested_fix=(
                    "Review the remaining TODO/FIXME markers and either resolve "
                    "them or track them explicitly."
                ),
            )
        )

    summary = ReviewWorkerSummary(
        agent_name=WorkerRole.STATIC_REVIEWER,
        summary=f"Static review produced {len(findings)} findings.",
        commands_run=commands,
        artifacts=artifacts,
    )
    return WorkerExecution(findings=findings, summary=summary)


async def _run_runtime_tester(
    run_id: str,
    sandbox_service: SandboxService,
    repo_map: RepoMapSummary,
) -> WorkerExecution:
    findings: list[ReviewFinding] = []
    artifacts: list[str] = []
    commands: list[str] = []

    if "python" in repo_map.languages:
        compile_response = await sandbox_service.execute_shell(
            run_id,
            _shell_request(WorkerRole.RUNTIME_TESTER, "python3 -m compileall ."),
        )
        commands.append(compile_response.command)
        artifacts.append(compile_response.artifact_path)

        if compile_response.exit_code != 0:
            findings.append(
                ReviewFinding(
                    title="Python compilation checks failed",
                    summary="`python3 -m compileall .` returned a non-zero exit code.",
                    severity=FindingSeverity.HIGH,
                    confidence=0.82,
                    source_agents=[WorkerRole.RUNTIME_TESTER],
                    evidence=(
                        _non_empty_lines(compile_response.stderr)[:10]
                        or _non_empty_lines(compile_response.stdout)[:10]
                    ),
                    suggested_fix=(
                        "Fix syntax and import-time issues until "
                        "`python3 -m compileall .` succeeds."
                    ),
                )
            )

        if repo_map.test_locations:
            pytest_probe = await sandbox_service.execute_shell(
                run_id,
                _shell_request(WorkerRole.RUNTIME_TESTER, "python3 -c 'import pytest'"),
            )
            commands.append(pytest_probe.command)
            artifacts.append(pytest_probe.artifact_path)

            if pytest_probe.exit_code == 0:
                pytest_response = await sandbox_service.execute_shell(
                    run_id,
                    _shell_request(
                        WorkerRole.RUNTIME_TESTER,
                        "python3 -m pytest -q",
                        timeout_seconds=180,
                    ),
                )
                commands.append(pytest_response.command)
                artifacts.append(pytest_response.artifact_path)

                if pytest_response.exit_code != 0:
                    findings.append(
                        ReviewFinding(
                            title="Automated tests did not pass in the sandbox",
                            summary="`python3 -m pytest -q` failed or reported test failures.",
                            severity=FindingSeverity.MEDIUM,
                            confidence=0.7,
                            source_agents=[WorkerRole.RUNTIME_TESTER],
                            evidence=(
                                _non_empty_lines(pytest_response.stderr)[:10]
                                or _non_empty_lines(pytest_response.stdout)[:10]
                            ),
                            reproduction_steps=["Run `python3 -m pytest -q` from the repo root."],
                            suggested_fix=(
                                "Stabilize the test suite or document the "
                                "environment/setup requirements clearly."
                            ),
                        )
                    )
            else:
                findings.append(
                    ReviewFinding(
                        title="Sandbox lacks pytest for runtime validation",
                        summary=(
                            "The runtime worker found tests but could not import "
                            "`pytest` inside the sandbox."
                        ),
                        severity=FindingSeverity.LOW,
                        confidence=0.61,
                        source_agents=[WorkerRole.RUNTIME_TESTER],
                        evidence=_non_empty_lines(pytest_probe.stderr)[:5],
                        suggested_fix=(
                            "Bundle test dependencies or document how the sandbox "
                            "should bootstrap them before execution."
                        ),
                    )
                )

    summary = ReviewWorkerSummary(
        agent_name=WorkerRole.RUNTIME_TESTER,
        summary=f"Runtime checks produced {len(findings)} findings.",
        commands_run=commands,
        artifacts=artifacts,
    )
    return WorkerExecution(findings=findings, summary=summary)


async def _run_security_reviewer(run_id: str, sandbox_service: SandboxService) -> WorkerExecution:
    findings: list[ReviewFinding] = []
    artifacts: list[str] = []
    commands: list[str] = []

    secret_response = await sandbox_service.execute_shell(
        run_id,
        _shell_request(
            WorkerRole.SECURITY_REVIEWER,
            command=(
                "rg --hidden -n -i "
                '"(begin [a-z ]*private key|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9_]{20,}|'
                "github_pat_[A-Za-z0-9_]{20,}|nvapi-[A-Za-z0-9_-]{20,}|"
                "sk-[A-Za-z0-9]{20,}|(api[_-]?key|secret|token|password)[[:space:]]*[:=]"
                "[[:space:]]*['\\\"]?[A-Za-z0-9_./+=-]{20,})\" "
                f". {RG_EXCLUDE_ARGS} || true"
            ),
        ),
    )
    commands.append(secret_response.command)
    artifacts.append(secret_response.artifact_path)

    secret_hits = _filter_secret_hits(_non_empty_lines(secret_response.stdout))
    if secret_hits:
        findings.append(
            ReviewFinding(
                title="Potential secret material detected in the repository",
                summary="Secret-like tokens or credentials were matched by the security scan.",
                severity=FindingSeverity.HIGH,
                confidence=0.78,
                source_agents=[WorkerRole.SECURITY_REVIEWER],
                evidence=secret_hits[:10],
                file_paths=_unique_paths(secret_hits),
                suggested_fix=(
                    "Rotate exposed credentials and move secrets to environment "
                    "or secret-management tooling."
                ),
            )
        )

    risky_response = await sandbox_service.execute_shell(
        run_id,
        _shell_request(
            WorkerRole.SECURITY_REVIEWER,
            command=(
                'rg -n "\\beval\\(|\\bexec\\(|shell=True|os\\.system\\(|'
                'pickle\\.loads\\(|yaml\\.load\\(" '
                f". {RG_EXCLUDE_ARGS} || true"
            ),
        ),
    )
    commands.append(risky_response.command)
    artifacts.append(risky_response.artifact_path)

    risky_hits = _filter_risky_hits(_non_empty_lines(risky_response.stdout))
    if risky_hits:
        findings.append(
            ReviewFinding(
                title="Potentially unsafe execution or deserialization patterns detected",
                summary=(
                    "The security scan found dynamic execution or unsafe "
                    "deserialization primitives."
                ),
                severity=FindingSeverity.MEDIUM,
                confidence=0.74,
                source_agents=[WorkerRole.SECURITY_REVIEWER],
                evidence=risky_hits[:10],
                file_paths=_unique_paths(risky_hits),
                suggested_fix=(
                    "Review each match and replace risky primitives with safer "
                    "alternatives where possible."
                ),
            )
        )

    summary = ReviewWorkerSummary(
        agent_name=WorkerRole.SECURITY_REVIEWER,
        summary=f"Security review produced {len(findings)} findings.",
        commands_run=commands,
        artifacts=artifacts,
    )
    return WorkerExecution(findings=findings, summary=summary)


async def _run_docs_devex_reviewer(
    run_id: str,
    sandbox_service: SandboxService,
    repo_map: RepoMapSummary,
) -> WorkerExecution:
    findings: list[ReviewFinding] = []
    artifacts: list[str] = []
    commands: list[str] = []

    if _repo_map_is_empty(repo_map):
        summary = ReviewWorkerSummary(
            agent_name=WorkerRole.DOCS_DEVEX_REVIEWER,
            summary="Docs/devex review skipped because the repository map was empty or invalid.",
            commands_run=commands,
            artifacts=artifacts,
        )
        return WorkerExecution(findings=findings, summary=summary)

    docs_probe = await sandbox_service.execute_shell(
        run_id,
        _shell_request(
            WorkerRole.DOCS_DEVEX_REVIEWER,
            command=(
                "find . -maxdepth 3 \\( -iname 'README*' -o -path './docs/*' "
                "-o -name '.env.example' \\) -print | sort | sed -n '1,120p'"
            ),
        ),
    )
    commands.append(docs_probe.command)
    artifacts.append(docs_probe.artifact_path)

    readme_path = next(
        (
            path
            for path in repo_map.docs_files
            if PurePosixPath(path).name.lower() in README_NAMES
        ),
        None,
    )
    if readme_path is None:
        findings.append(
            ReviewFinding(
                title="README is missing",
                summary=(
                    "The repository does not appear to include a README file at "
                    "the scanned depth."
                ),
                severity=FindingSeverity.MEDIUM,
                confidence=0.86,
                source_agents=[WorkerRole.DOCS_DEVEX_REVIEWER],
                suggested_fix=(
                    "Add a README with project purpose, setup, run, and test "
                    "instructions."
                ),
            )
        )
    else:
        readme_content = (
            await sandbox_service.read_file(run_id, readme_path.removeprefix("./"))
        ).content
        missing_sections = [
            section
            for section in ("install", "setup", "run", "test")
            if section not in readme_content.lower()
        ]
        if missing_sections:
            findings.append(
                ReviewFinding(
                    title="README is present but missing common onboarding sections",
                    summary=f"The README does not mention: {', '.join(missing_sections)}.",
                    severity=FindingSeverity.LOW,
                    confidence=0.67,
                    source_agents=[WorkerRole.DOCS_DEVEX_REVIEWER],
                    evidence=[readme_path],
                    file_paths=[readme_path],
                    suggested_fix=(
                        "Add clear install, setup, run, and test guidance to "
                        "reduce onboarding friction."
                    ),
                )
            )

    if not any(
        PurePosixPath(path).name == ".env.example"
        for path in repo_map.top_level_files + repo_map.docs_files
    ):
        findings.append(
            ReviewFinding(
                title="Environment example file is missing",
                summary="No `.env.example` file was found in the scanned repository view.",
                severity=FindingSeverity.LOW,
                confidence=0.7,
                source_agents=[WorkerRole.DOCS_DEVEX_REVIEWER],
                suggested_fix=(
                    "Add a `.env.example` file that documents required "
                    "environment variables without real secrets."
                ),
            )
        )

    if not repo_map.test_locations:
        findings.append(
            ReviewFinding(
                title="Tests are not obvious from the repository structure",
                summary=(
                    "The review could not detect a test directory or test file "
                    "path in the scanned repository view."
                ),
                severity=FindingSeverity.MEDIUM,
                confidence=0.63,
                source_agents=[WorkerRole.DOCS_DEVEX_REVIEWER],
                suggested_fix=(
                    "Add automated tests or make the existing test locations "
                    "easier to discover."
                ),
            )
        )

    summary = ReviewWorkerSummary(
        agent_name=WorkerRole.DOCS_DEVEX_REVIEWER,
        summary=f"Docs/devex review produced {len(findings)} findings.",
        commands_run=commands,
        artifacts=artifacts,
    )
    return WorkerExecution(findings=findings, summary=summary)


async def _run_external_validator(
    run_id: str,
    sandbox_service: SandboxService,
    repo_map: RepoMapSummary,
) -> WorkerExecution:
    findings: list[ReviewFinding] = []
    commands: list[str] = []
    artifacts: list[str] = []

    dependency_probe = await sandbox_service.execute_shell(
        run_id,
        _shell_request(
            WorkerRole.EXTERNAL_VALIDATOR,
            command=(
                "find . -maxdepth 4 \\( -name requirements.txt -o -name pyproject.toml "
                "-o -name package.json -o -name poetry.lock -o -name package-lock.json "
                "-o -name pnpm-lock.yaml \\) -print | sort | sed -n '1,120p'"
            ),
        ),
    )
    commands.append(dependency_probe.command)
    artifacts.append(dependency_probe.artifact_path)

    if "requirements.txt" in {PurePosixPath(path).name for path in repo_map.dependency_files}:
        requirements_content = (
            await sandbox_service.read_file(run_id, "requirements.txt")
        ).content
        unpinned = [
            line.strip()
            for line in requirements_content.splitlines()
            if line.strip() and not line.strip().startswith("#") and "==" not in line
        ]
        if unpinned:
            findings.append(
                ReviewFinding(
                    title="Some Python dependencies are not pinned exactly",
                    summary="The requirements file includes dependencies without `==` pinning.",
                    severity=FindingSeverity.LOW,
                    confidence=0.68,
                    source_agents=[WorkerRole.EXTERNAL_VALIDATOR],
                    evidence=unpinned[:10],
                    file_paths=["requirements.txt"],
                    suggested_fix=(
                        "Pin production-critical dependencies or document the "
                        "accepted version range policy."
                    ),
                )
            )

    if "package.json" in {PurePosixPath(path).name for path in repo_map.dependency_files}:
        package_json = (
            await sandbox_service.read_file(run_id, "package.json")
        ).content
        has_frontend_framework = any(
            token in package_json for token in ('"react"', '"next"', '"vite"')
        )
        if has_frontend_framework and "playwright" not in package_json.lower():
            findings.append(
                ReviewFinding(
                    title="Frontend project detected without browser validation tooling",
                    summary=(
                        "The manifest suggests a web UI, but Playwright was not "
                        "detected in `package.json`."
                    ),
                    severity=FindingSeverity.LOW,
                    confidence=0.62,
                    source_agents=[WorkerRole.EXTERNAL_VALIDATOR],
                    file_paths=["package.json"],
                    suggested_fix=(
                        "Add Playwright or another browser-level validation path "
                        "for UI regressions."
                    ),
                )
            )

    summary = ReviewWorkerSummary(
        agent_name=WorkerRole.EXTERNAL_VALIDATOR,
        summary=f"External validation produced {len(findings)} findings.",
        commands_run=commands,
        artifacts=artifacts,
    )
    return WorkerExecution(findings=findings, summary=summary)


def _detect_languages(file_path: str) -> list[str]:
    suffix = PurePosixPath(file_path).suffix.lower()
    language = CODE_EXTENSIONS.get(suffix)
    return [language] if language else []


def _parse_wc_output(output: str) -> list[tuple[int, str]]:
    results: list[tuple[int, str]] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.endswith("total"):
            continue
        match = re.match(r"^(?P<count>\d+)\s+(?P<path>.+)$", stripped)
        if not match:
            continue
        results.append((int(match.group("count")), match.group("path")))
    return results


def _non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _filter_secret_hits(lines: list[str]) -> list[str]:
    hits: list[str] = []
    for line in lines:
        path = _extract_rg_path(line)
        if path and _is_low_signal_security_path(path):
            continue
        if not _contains_real_secret_value(line):
            continue
        hits.append(_redact_secret_values(line))
    return hits


def _contains_real_secret_value(line: str) -> bool:
    for pattern in SECRET_VALUE_PATTERNS:
        match = pattern.search(line)
        if not match:
            continue
        value = match.groupdict().get("value")
        if value and _is_non_secret_assignment_value(value):
            continue
        return True
    return False


def _is_non_secret_assignment_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"your_key_here", "placeholder", "changeme", "example", "none", "null"}:
        return True
    return normalized.startswith(("settings.", "self.", "config.", "os.environ", "getenv"))


def _filter_risky_hits(lines: list[str]) -> list[str]:
    hits: list[str] = []
    for line in lines:
        path = _extract_rg_path(line)
        if path and _is_low_signal_security_path(path):
            continue
        if "rg -n" in line or "rg --hidden" in line:
            continue
        hits.append(line)
    return hits


def _unique_paths(lines: list[str]) -> list[str]:
    paths: list[str] = []
    for line in lines:
        path = _extract_rg_path(line)
        if path and path not in paths:
            paths.append(path)
    return paths[:10]


def _extract_rg_path(line: str) -> str | None:
    if ":" not in line:
        return None
    return line.split(":", maxsplit=1)[0]


def _is_low_signal_security_path(path: str) -> bool:
    normalized = path.removeprefix("./").lower()
    if normalized.startswith(("tests/", "docs/", "ui/dist/")):
        return True
    return PurePosixPath(normalized).name in {
        ".env.example",
        "readme.md",
        "demo.md",
        "package-lock.json",
    }


def _redact_secret_values(line: str) -> str:
    redacted = line
    for pattern in SECRET_VALUE_PATTERNS:
        redacted = pattern.sub(_redact_match, redacted)
    return redacted


def _redact_match(match: re.Match[str]) -> str:
    value = match.groupdict().get("value")
    if not value:
        return "<redacted-secret>"
    return match.group(0).replace(value, "<redacted-secret>")


def _repo_map_is_empty(repo_map: RepoMapSummary) -> bool:
    if repo_map.mapped_file_count > 0:
        return False
    return not any(
        (
            repo_map.top_level_files,
            repo_map.languages,
            repo_map.dependency_files,
            repo_map.entry_points,
            repo_map.test_locations,
            repo_map.docs_files,
        )
    )


def _trim_scan_stderr(text: str, limit: int = 240) -> str | None:
    cleaned = " ".join(_non_empty_lines(text))
    if not cleaned:
        return None
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3]}..."
