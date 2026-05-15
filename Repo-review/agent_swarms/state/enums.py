"""Shared enums for the application."""

from enum import Enum


class RunMode(str, Enum):
    """Supported run modes."""

    CHAT = "chat"
    REVIEW = "review"


class ReviewIntent(str, Enum):
    """Execution intent for repository runs."""

    REVIEW = "review"
    FIX = "fix"


class RunStatus(str, Enum):
    """High-level run lifecycle."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class RunEventType(str, Enum):
    """Structured lifecycle events emitted during review runs."""

    RUN_CREATED = "run_created"
    RUN_STARTED = "run_started"
    SANDBOX_READY = "sandbox_ready"
    PLAN_READY = "plan_ready"
    WORKER_STARTED = "worker_started"
    WORKER_COMPLETED = "worker_completed"
    TOOL_COMPLETED = "tool_completed"
    FIX_STARTED = "fix_started"
    FILE_UPDATED = "file_updated"
    FIX_COMPLETED = "fix_completed"
    REPORT_READY = "report_ready"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


class ProviderBackend(str, Enum):
    """Supported LLM backends."""

    MOCK = "mock"
    NVIDIA = "nvidia"


class SandboxBackend(str, Enum):
    """Supported sandbox backends."""

    LOCAL = "local"
    DOCKER = "docker"
    OPENSANDBOX = "opensandbox"


class RepoSourceType(str, Enum):
    """Supported repository source types."""

    LOCAL = "local"
    REMOTE = "remote"


class SandboxRunStatus(str, Enum):
    """Sandbox run lifecycle."""

    READY = "ready"
    CLOSED = "closed"
    FAILED = "failed"


class WorkerRole(str, Enum):
    """Supported review worker roles."""

    REPO_MAPPER = "repo_mapper"
    STATIC_REVIEWER = "static_reviewer"
    RUNTIME_TESTER = "runtime_tester"
    SECURITY_REVIEWER = "security_reviewer"
    DOCS_DEVEX_REVIEWER = "docs_devex_reviewer"
    EXTERNAL_VALIDATOR = "external_validator"


class FindingSeverity(str, Enum):
    """Supported review finding severities."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewVerdict(str, Enum):
    """Overall review verdict."""

    GREEN = "green"
    CAUTION = "caution"
    RED = "red"
