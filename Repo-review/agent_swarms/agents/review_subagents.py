"""LLM-assisted review sub-agents that use deterministic sandbox tools."""

from __future__ import annotations

from dataclasses import dataclass
import json

from agent_swarms.agents.review_workers import WorkerExecution
from agent_swarms.agents.review_tool_registry import tool_for_worker
from agent_swarms.config import Settings
from agent_swarms.state.enums import ProviderBackend, WorkerRole
from agent_swarms.state.schemas import RepoMapSummary


SUB_AGENT_SYSTEM_PROMPT = """
You are a repository review sub-agent.
Write in first person as the assigned worker.
Use short, plain-language status messages.
Do not invent files, findings, commands, or fixes.
Only summarize the evidence provided by the sandbox tool.
""".strip()


@dataclass(frozen=True)
class ReviewAgentText:
    """One generated sub-agent message."""

    text: str
    backend: ProviderBackend
    used_llm: bool
    error: str | None = None


class ReviewSubAgentMessenger:
    """Generate concise sub-agent plan and result messages."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = self._build_client(settings)

    async def plan_tool_use(
        self,
        *,
        worker_role: WorkerRole,
        user_query: str,
        repo_map: RepoMapSummary | None = None,
    ) -> ReviewAgentText:
        """Return the sub-agent's plan before tool execution."""

        tool = tool_for_worker(worker_role)
        fallback = (
            f"I will handle the {_role_label(worker_role)} step and use the "
            f"`{tool.name}` sandbox tool to collect evidence."
        )
        prompt = {
            "task": "Write one short sentence before tool execution.",
            "worker_role": worker_role.value,
            "user_query": user_query,
            "available_tool": {
                "name": tool.name,
                "description": tool.description,
            },
            "repo_map": _repo_map_brief(repo_map),
        }
        return await self._generate(json.dumps(prompt, sort_keys=True), fallback)

    async def summarize_tool_result(
        self,
        *,
        worker_role: WorkerRole,
        execution: WorkerExecution,
    ) -> ReviewAgentText:
        """Return the sub-agent's final message after tool execution."""

        findings = [
            {
                "severity": finding.severity.value,
                "title": finding.title,
                "summary": finding.summary,
                "file_paths": finding.file_paths[:3],
            }
            for finding in execution.findings[:5]
        ]
        fallback = _fallback_final_message(worker_role, execution)
        prompt = {
            "task": "Write one or two short sentences after tool execution.",
            "worker_role": worker_role.value,
            "tool": tool_for_worker(worker_role).name,
            "findings_count": len(execution.findings),
            "commands_run": execution.summary.commands_run[:5],
            "findings": findings,
            "summary": execution.summary.summary,
        }
        return await self._generate(json.dumps(prompt, sort_keys=True), fallback)

    def _build_client(self, settings: Settings):
        if settings.llm_backend is not ProviderBackend.NVIDIA or not settings.nvidia_api_key:
            return None

        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except ImportError:
            return None

        return ChatNVIDIA(
            nvidia_api_key=settings.nvidia_api_key,
            model=settings.nvidia_model,
            temperature=0,
        )

    async def _generate(self, user_prompt: str, fallback: str) -> ReviewAgentText:
        if self._client is None:
            return ReviewAgentText(
                text=fallback,
                backend=ProviderBackend.MOCK,
                used_llm=False,
            )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            response = await self._client.ainvoke(
                [
                    SystemMessage(content=SUB_AGENT_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
        except Exception as exc:
            return ReviewAgentText(
                text=fallback,
                backend=self._settings.llm_backend,
                used_llm=False,
                error=str(exc),
            )

        content = response.content
        if isinstance(content, str):
            text = content.strip()
        else:
            text = " ".join(str(part) for part in content).strip()

        return ReviewAgentText(
            text=text or fallback,
            backend=self._settings.llm_backend,
            used_llm=True,
        )


def _fallback_final_message(worker_role: WorkerRole, execution: WorkerExecution) -> str:
    finding_count = len(execution.findings)
    if finding_count == 0:
        return (
            f"I finished the {_role_label(worker_role)} check and did not find "
            "issues in this scope."
        )
    return (
        f"I finished the {_role_label(worker_role)} check and found "
        f"{finding_count} issue{'' if finding_count == 1 else 's'} to review."
    )


def _repo_map_brief(repo_map: RepoMapSummary | None) -> dict[str, object]:
    if repo_map is None:
        return {}
    return {
        "mapped_file_count": repo_map.mapped_file_count,
        "languages": repo_map.languages,
        "dependency_files": repo_map.dependency_files[:5],
        "entry_points": repo_map.entry_points[:5],
        "test_locations": repo_map.test_locations[:5],
        "docs_files": repo_map.docs_files[:5],
    }


def _role_label(worker_role: WorkerRole) -> str:
    return worker_role.value.replace("_", " ")
