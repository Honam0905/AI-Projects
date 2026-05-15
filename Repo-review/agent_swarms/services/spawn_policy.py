"""Spawn policy for repository reviews."""

from agent_swarms.state.enums import ReviewIntent, WorkerRole
from agent_swarms.state.schemas import ReviewPlan

ALL_WORKERS = [
    WorkerRole.REPO_MAPPER,
    WorkerRole.STATIC_REVIEWER,
    WorkerRole.RUNTIME_TESTER,
    WorkerRole.SECURITY_REVIEWER,
    WorkerRole.DOCS_DEVEX_REVIEWER,
    WorkerRole.EXTERNAL_VALIDATOR,
]
PRODUCTION_REVIEW_HINTS = (
    "production-ready",
    "full review",
    "audit",
    "full repo",
    "full repository",
)
SECURITY_HINTS = ("security", "secret", "vulnerability")
DOCS_HINTS = ("docs", "readme", "onboarding", "devex")
RUNTIME_HINTS = ("runtime", "test", "build")
EXTERNAL_VALIDATION_HINTS = (
    "dependency",
    "version",
    "frontend",
    "browser",
)
STATIC_REVIEW_HINTS = ("architecture", "maintainability", "refactor")


def build_review_plan(
    user_query: str,
    *,
    intent: ReviewIntent = ReviewIntent.REVIEW,
) -> ReviewPlan:
    """Build a deterministic review plan from the user query."""

    query = user_query.lower()

    if any(token in query for token in PRODUCTION_REVIEW_HINTS):
        workers = ALL_WORKERS
        reason = "broad repository audit requested; use the full review swarm"
    elif any(token in query for token in SECURITY_HINTS):
        workers = [WorkerRole.REPO_MAPPER, WorkerRole.SECURITY_REVIEWER]
        reason = (
            "security-focused review requested; prioritize repository mapping "
            "plus security checks"
        )
    elif any(token in query for token in DOCS_HINTS):
        workers = [WorkerRole.REPO_MAPPER, WorkerRole.DOCS_DEVEX_REVIEWER]
        reason = (
            "documentation-focused review requested; prioritize repository "
            "mapping plus docs/devex checks"
        )
    elif any(token in query for token in RUNTIME_HINTS):
        workers = [WorkerRole.REPO_MAPPER, WorkerRole.RUNTIME_TESTER]
        reason = (
            "runtime-focused review requested; prioritize repository mapping "
            "plus execution checks"
        )
    elif any(token in query for token in EXTERNAL_VALIDATION_HINTS):
        workers = [WorkerRole.REPO_MAPPER, WorkerRole.EXTERNAL_VALIDATOR]
        reason = (
            "dependency or validation-focused review requested; prioritize "
            "repository mapping plus external validation"
        )
    elif any(token in query for token in STATIC_REVIEW_HINTS):
        workers = [WorkerRole.REPO_MAPPER, WorkerRole.STATIC_REVIEWER]
        reason = (
            "code-structure review requested; prioritize repository mapping "
            "plus static analysis"
        )
    else:
        workers = [
            WorkerRole.REPO_MAPPER,
            WorkerRole.STATIC_REVIEWER,
            WorkerRole.SECURITY_REVIEWER,
            WorkerRole.DOCS_DEVEX_REVIEWER,
        ]
        reason = "general repository review requested; use the core review swarm"

    return ReviewPlan(
        intent=intent,
        spawn_count=len(workers),
        selected_workers=workers,
        route_reason=reason,
        expected_outputs=[
            "repo_map",
            "findings",
            "worker_summaries",
            "commands_run",
            "artifacts",
            *(["fix_result"] if intent is ReviewIntent.FIX else []),
        ],
    )
