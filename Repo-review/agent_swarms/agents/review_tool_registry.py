"""Tool metadata for review sub-agents."""

from __future__ import annotations

from dataclasses import dataclass

from agent_swarms.state.enums import WorkerRole


@dataclass(frozen=True)
class ReviewAgentTool:
    """Tool metadata exposed to review sub-agents and event streams."""

    name: str
    description: str


def tool_for_worker(worker_role: WorkerRole) -> ReviewAgentTool:
    """Return the deterministic sandbox tool used by a sub-agent."""

    tools = {
        WorkerRole.REPO_MAPPER: ReviewAgentTool(
            name="repo_map_shell_tool",
            description="Runs sandbox shell commands to list files and detect repo structure.",
        ),
        WorkerRole.STATIC_REVIEWER: ReviewAgentTool(
            name="static_review_shell_tool",
            description="Runs sandbox shell commands to inspect file size and TODO/FIXME markers.",
        ),
        WorkerRole.RUNTIME_TESTER: ReviewAgentTool(
            name="runtime_test_shell_tool",
            description="Runs sandbox shell commands for compile and test checks.",
        ),
        WorkerRole.SECURITY_REVIEWER: ReviewAgentTool(
            name="security_scan_shell_tool",
            description="Runs sandbox shell scans for secrets and risky execution patterns.",
        ),
        WorkerRole.DOCS_DEVEX_REVIEWER: ReviewAgentTool(
            name="docs_devex_sandbox_tool",
            description=(
                "Uses sandbox file and shell evidence to inspect README, setup, "
                "and test guidance."
            ),
        ),
        WorkerRole.EXTERNAL_VALIDATOR: ReviewAgentTool(
            name="dependency_validation_sandbox_tool",
            description=(
                "Uses sandbox file and shell evidence to inspect dependency "
                "and validation signals."
            ),
        ),
    }
    return tools[worker_role]
