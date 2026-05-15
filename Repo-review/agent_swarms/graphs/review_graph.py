"""LangGraph workflow for repository review swarms."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent_swarms.agents.review_subagents import ReviewSubAgentMessenger
from agent_swarms.agents.review_tool_registry import tool_for_worker
from agent_swarms.agents.review_workers import run_repo_mapper, run_specialist
from agent_swarms.config import Settings
from agent_swarms.sandbox.service import SandboxService
from agent_swarms.services.deduplication import deduplicate_findings
from agent_swarms.services.report_builder import build_review_report
from agent_swarms.services.review_run_tracking import ReviewEventBroker
from agent_swarms.services.spawn_policy import build_review_plan
from agent_swarms.state.enums import RunEventType, RunStatus, WorkerRole
from agent_swarms.state.schemas import (
    RepoMapSummary,
    ReviewFinding,
    ReviewPlan,
    ReviewGraphState,
    ReviewWorkerSummary,
)


def build_review_graph(
    sandbox_service: SandboxService,
    event_broker: ReviewEventBroker | None = None,
    settings: Settings | None = None,
):
    """Compile the repository review graph."""

    sub_agent_messenger = ReviewSubAgentMessenger(settings or Settings())

    async def publish_event(
        *,
        run_id: str,
        event_type: RunEventType,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Persist a review lifecycle event when streaming is enabled."""

        if event_broker is None:
            return
        await event_broker.publish(
            run_id=run_id,
            event_type=event_type,
            message=message,
            status=RunStatus.IN_PROGRESS,
            payload=payload,
        )

    async def classify_review(state: ReviewGraphState) -> ReviewGraphState:
        """Attach or build the deterministic worker plan."""

        if "plan" in state and state["plan"]:
            plan = ReviewPlan.model_validate(state["plan"])
        else:
            plan = build_review_plan(state["user_query"])
        return {"plan": plan.model_dump()}

    async def repo_mapper(state: ReviewGraphState) -> ReviewGraphState:
        """Map the repository before specialist workers run."""

        agent_plan = await sub_agent_messenger.plan_tool_use(
            worker_role=WorkerRole.REPO_MAPPER,
            user_query=state["user_query"],
        )
        mapper_tool = tool_for_worker(WorkerRole.REPO_MAPPER)
        await publish_event(
            run_id=state["run_id"],
            event_type=RunEventType.WORKER_STARTED,
            message=agent_plan.text,
            payload={
                "worker_role": WorkerRole.REPO_MAPPER.value,
                "agent_message": agent_plan.text,
                "llm_backend": agent_plan.backend.value,
                "llm_used": agent_plan.used_llm,
                "llm_error": agent_plan.error,
                "tool_name": mapper_tool.name,
            },
        )
        execution = await run_repo_mapper(state["run_id"], sandbox_service)
        agent_final = await sub_agent_messenger.summarize_tool_result(
            worker_role=WorkerRole.REPO_MAPPER,
            execution=execution,
        )
        summary = execution.summary.model_copy(update={"summary": agent_final.text})
        await publish_event(
            run_id=state["run_id"],
            event_type=RunEventType.WORKER_COMPLETED,
            message=agent_final.text,
            payload={
                "worker_role": WorkerRole.REPO_MAPPER.value,
                "agent_message": agent_final.text,
                "llm_backend": agent_final.backend.value,
                "llm_used": agent_final.used_llm,
                "llm_error": agent_final.error,
                "tool_name": mapper_tool.name,
                "mapped_file_count": (
                    execution.repo_map.mapped_file_count
                    if execution.repo_map
                    else 0
                ),
                "languages": execution.repo_map.languages if execution.repo_map else [],
                "scan_stderr": execution.repo_map.scan_stderr if execution.repo_map else None,
                "artifacts": execution.summary.artifacts,
                "commands_run": execution.summary.commands_run,
            },
        )
        return {
            "repo_map": execution.repo_map.model_dump() if execution.repo_map else {},
            "worker_summaries": [summary.model_dump()],
            "commands_run": execution.summary.commands_run,
            "artifacts": execution.summary.artifacts,
        }

    def dispatch_specialists(state: ReviewGraphState) -> str | list[Send]:
        """Fan out selected specialist workers in parallel."""

        selected_workers = [
            role if isinstance(role, WorkerRole) else WorkerRole(role)
            for role in state["plan"]["selected_workers"]
        ]
        specialist_roles = [role for role in selected_workers if role is not WorkerRole.REPO_MAPPER]
        if not specialist_roles:
            return "synthesize_report"
        return [
            Send(
                "worker_node",
                {
                    "run_id": state["run_id"],
                    "user_query": state["user_query"],
                    "worker_role": role,
                    "repo_map": state["repo_map"],
                },
            )
            for role in specialist_roles
        ]

    async def worker_node(state: dict[str, Any]) -> ReviewGraphState:
        """Run one specialist worker and publish its tool-backed result."""

        role = state["worker_role"]
        worker_role = role if isinstance(role, WorkerRole) else WorkerRole(role)
        repo_map = RepoMapSummary.model_validate(state["repo_map"])
        agent_plan = await sub_agent_messenger.plan_tool_use(
            worker_role=worker_role,
            user_query=state.get("user_query", ""),
            repo_map=repo_map,
        )
        worker_tool = tool_for_worker(worker_role)
        await publish_event(
            run_id=state["run_id"],
            event_type=RunEventType.WORKER_STARTED,
            message=agent_plan.text,
            payload={
                "worker_role": worker_role.value,
                "agent_message": agent_plan.text,
                "llm_backend": agent_plan.backend.value,
                "llm_used": agent_plan.used_llm,
                "llm_error": agent_plan.error,
                "tool_name": worker_tool.name,
            },
        )
        execution = await run_specialist(
            worker_role=worker_role,
            run_id=state["run_id"],
            sandbox_service=sandbox_service,
            repo_map=repo_map,
        )
        agent_final = await sub_agent_messenger.summarize_tool_result(
            worker_role=worker_role,
            execution=execution,
        )
        summary = execution.summary.model_copy(update={"summary": agent_final.text})
        await publish_event(
            run_id=state["run_id"],
            event_type=RunEventType.WORKER_COMPLETED,
            message=agent_final.text,
            payload={
                "worker_role": worker_role.value,
                "agent_message": agent_final.text,
                "llm_backend": agent_final.backend.value,
                "llm_used": agent_final.used_llm,
                "llm_error": agent_final.error,
                "tool_name": worker_tool.name,
                "findings_count": len(execution.findings),
                "artifacts": execution.summary.artifacts,
                "commands_run": execution.summary.commands_run,
            },
        )
        return {
            "findings": [finding.model_dump() for finding in execution.findings],
            "worker_summaries": [summary.model_dump()],
            "commands_run": execution.summary.commands_run,
            "artifacts": execution.summary.artifacts,
        }

    async def synthesize_report(state: ReviewGraphState) -> ReviewGraphState:
        findings = [
            ReviewFinding.model_validate(item)
            for item in state.get("findings", [])
        ]
        deduped_findings = deduplicate_findings(findings)
        worker_summaries = [
            ReviewWorkerSummary.model_validate(item)
            for item in state.get("worker_summaries", [])
        ]
        repo_map = RepoMapSummary.model_validate(state["repo_map"])
        report = build_review_report(
            repo_map=repo_map,
            findings=deduped_findings,
            worker_summaries=worker_summaries,
            commands_run=state.get("commands_run", []),
        )
        await publish_event(
            run_id=state["run_id"],
            event_type=RunEventType.REPORT_READY,
            message="Final review report is ready.",
            payload={
                "overall_verdict": report.overall_verdict.value,
                "top_findings_count": len(report.top_findings),
            },
        )
        return {
            "findings": [finding.model_dump() for finding in deduped_findings],
            "final_report": report.model_dump(),
        }

    workflow = StateGraph(ReviewGraphState)
    workflow.add_node("classify_review", classify_review)
    workflow.add_node("repo_mapper", repo_mapper)
    workflow.add_node("worker_node", worker_node)
    workflow.add_node("synthesize_report", synthesize_report)
    workflow.add_edge(START, "classify_review")
    workflow.add_edge("classify_review", "repo_mapper")
    workflow.add_conditional_edges("repo_mapper", dispatch_specialists)
    workflow.add_edge("worker_node", "synthesize_report")
    workflow.add_edge("synthesize_report", END)
    return workflow.compile()
