"""Typed contracts for the API and graph."""

from datetime import datetime
from operator import add
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field, model_validator

from agent_swarms.state.enums import (
    FindingSeverity,
    ProviderBackend,
    RepoSourceType,
    ReviewIntent,
    RunEventType,
    ReviewVerdict,
    RunMode,
    RunStatus,
    SandboxBackend,
    SandboxRunStatus,
    WorkerRole,
)


class ChatRequest(BaseModel):
    """Incoming chat request."""

    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=128)
    requested_mode: RunMode | None = None


class RunPlan(BaseModel):
    """Minimal execution plan returned by the supervisor router."""

    mode: RunMode
    spawn_count: int = Field(default=0, ge=0, le=6)
    selected_workers: list[str] = Field(default_factory=list)
    needs_sandbox: bool = False
    route_reason: str


class ChatResponse(BaseModel):
    """Outgoing chat response."""

    request_id: str
    status: RunStatus
    provider: ProviderBackend
    plan: RunPlan
    response: str


class GraphState(TypedDict, total=False):
    """State carried through the supervisor routing workflow."""

    request_id: str
    session_id: str | None
    user_message: str
    requested_mode: RunMode | None
    mode: RunMode
    route_reason: str
    spawn_count: int
    selected_workers: list[str]
    needs_sandbox: bool
    provider_backend: ProviderBackend
    response: str


class SandboxRunCreateRequest(BaseModel):
    """Request to create a sandbox-backed run."""

    repo_path: str | None = Field(default=None, min_length=1)
    sandbox_backend: SandboxBackend | None = None


class SandboxRunResponse(BaseModel):
    """Sandbox run metadata returned to API clients."""

    run_id: str
    sandbox_backend: SandboxBackend
    status: SandboxRunStatus
    workspace_path: str
    repo_source_path: str | None = None
    sandbox_id: str | None = None
    created_at: datetime
    closed_at: datetime | None = None


class ShellExecutionRequest(BaseModel):
    """Request to run a shell command inside a sandbox."""

    command: str = Field(min_length=1, max_length=12000)
    timeout_seconds: int | None = Field(default=None, ge=1, le=900)
    worker_role: WorkerRole | None = None
    tool_name: str | None = Field(default=None, max_length=128)


class PythonExecutionRequest(BaseModel):
    """Request to run Python code inside a sandbox."""

    code: str = Field(min_length=1, max_length=20000)
    timeout_seconds: int | None = Field(default=None, ge=1, le=900)
    python_executable: str | None = Field(default=None, max_length=128)
    worker_role: WorkerRole | None = None
    tool_name: str | None = Field(default=None, max_length=128)


class SandboxExecutionResponse(BaseModel):
    """Normalized command or python execution result."""

    run_id: str
    sandbox_backend: SandboxBackend
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    artifact_path: str
    executed_at: datetime


class SandboxFileEntry(BaseModel):
    """File or directory inside the sandbox workspace."""

    path: str
    is_dir: bool
    size_bytes: int


class SandboxFileListResponse(BaseModel):
    """List of files inside the sandbox workspace."""

    run_id: str
    path: str
    entries: list[SandboxFileEntry]


class SandboxFileContentResponse(BaseModel):
    """File contents read from the sandbox workspace."""

    run_id: str
    path: str
    content: str


class SandboxArtifactEntry(BaseModel):
    """Artifact metadata persisted for a run."""

    path: str
    size_bytes: int
    created_at: datetime


class SandboxArtifactsResponse(BaseModel):
    """Artifact listing for a sandbox run."""

    run_id: str
    artifacts: list[SandboxArtifactEntry]


class ReviewRequest(BaseModel):
    """Incoming repository review request."""

    repo_path: str | None = Field(default=None, min_length=1)
    repo_url: str | None = Field(default=None, min_length=1, max_length=2048)
    user_query: str = Field(
        default="Review this repository carefully.",
        min_length=1,
        max_length=4000,
    )
    intent: ReviewIntent = ReviewIntent.REVIEW
    sandbox_backend: SandboxBackend | None = None
    session_id: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_repo_source(self) -> "ReviewRequest":
        """Reject ambiguous review requests."""

        if self.repo_path and self.repo_url:
            raise ValueError("Provide either repo_path or repo_url, not both.")
        return self


class ReviewPlan(BaseModel):
    """Structured review execution plan."""

    intent: ReviewIntent = ReviewIntent.REVIEW
    needs_repo: bool = True
    needs_sandbox: bool = True
    spawn_count: int = Field(ge=1, le=6)
    selected_workers: list[WorkerRole]
    route_reason: str
    expected_outputs: list[str]


class RepoMapSummary(BaseModel):
    """Repository structure summary."""

    mapped_file_count: int = 0
    top_level_files: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    dependency_files: list[str] = Field(default_factory=list)
    entry_points: list[str] = Field(default_factory=list)
    test_locations: list[str] = Field(default_factory=list)
    docs_files: list[str] = Field(default_factory=list)
    scan_stderr: str | None = None


class ReviewFinding(BaseModel):
    """One structured review finding."""

    title: str
    summary: str
    severity: FindingSeverity
    confidence: float = Field(ge=0, le=1)
    source_agents: list[WorkerRole]
    evidence: list[str] = Field(default_factory=list)
    file_paths: list[str] = Field(default_factory=list)
    reproduction_steps: list[str] = Field(default_factory=list)
    suggested_fix: str | None = None


class ReviewWorkerSummary(BaseModel):
    """Per-worker execution summary."""

    agent_name: WorkerRole
    summary: str
    commands_run: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)


class ReviewReport(BaseModel):
    """Structured final review report."""

    overall_verdict: ReviewVerdict
    summary: str
    repo_map: RepoMapSummary
    top_findings: list[ReviewFinding]
    worker_summaries: list[ReviewWorkerSummary]
    commands_run: list[str]
    artifacts: list[SandboxArtifactEntry] = Field(default_factory=list)


class FixResult(BaseModel):
    """Summary of automatic fixes applied inside the sandbox."""

    applied: bool = False
    summary: str
    changed_files: list[str] = Field(default_factory=list)
    fixed_finding_titles: list[str] = Field(default_factory=list)
    unsupported_finding_titles: list[str] = Field(default_factory=list)
    diff_artifact_path: str | None = None
    applied_to_local_repo: bool = False
    apply_back_target: str | None = None
    apply_back_error: str | None = None


class ReviewResponse(BaseModel):
    """Review response returned by the API."""

    run_id: str
    status: RunStatus
    intent: ReviewIntent
    repo_path: str
    repo_source_type: RepoSourceType
    repo_source: str
    sandbox_backend: SandboxBackend
    plan: ReviewPlan
    report: ReviewReport
    fix_result: FixResult | None = None


class ReviewRunSubmittedResponse(BaseModel):
    """Accepted async review submission."""

    run_id: str
    status: RunStatus
    intent: ReviewIntent
    repo_path: str
    repo_source_type: RepoSourceType
    repo_source: str
    sandbox_backend: SandboxBackend
    plan: ReviewPlan
    stream_path: str


class ReviewRunEvent(BaseModel):
    """One replayable lifecycle event for a review run."""

    event_id: str
    run_id: str
    event_type: RunEventType
    message: str
    timestamp: datetime
    status: RunStatus | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ReviewRunSummary(BaseModel):
    """Summary of a persisted review run."""

    run_id: str
    mode: RunMode = RunMode.REVIEW
    status: RunStatus
    intent: ReviewIntent = ReviewIntent.REVIEW
    repo_path: str
    repo_source_type: RepoSourceType = RepoSourceType.LOCAL
    repo_source: str
    user_query: str
    sandbox_backend: SandboxBackend
    plan: ReviewPlan | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None


class ReviewRunDetail(ReviewRunSummary):
    """Detailed persisted review run state."""

    report: ReviewReport | None = None
    artifacts: list[SandboxArtifactEntry] = Field(default_factory=list)
    fix_result: FixResult | None = None


class ReviewArtifactContentResponse(BaseModel):
    """Text content for a persisted review artifact."""

    run_id: str
    path: str
    content: str


class ReviewRunListResponse(BaseModel):
    """List of persisted review runs."""

    runs: list[ReviewRunSummary]


class ReviewGraphState(TypedDict, total=False):
    """State carried through the review graph."""

    run_id: str
    repo_path: str
    user_query: str
    sandbox_backend: SandboxBackend
    plan: dict[str, Any]
    repo_map: dict[str, Any]
    findings: Annotated[list[dict[str, Any]], add]
    worker_summaries: Annotated[list[dict[str, Any]], add]
    commands_run: Annotated[list[str], add]
    artifacts: Annotated[list[str], add]
    final_report: dict[str, Any]
