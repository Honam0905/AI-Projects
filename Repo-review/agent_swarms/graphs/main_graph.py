"""Supervisor routing graph for chat and repository review requests."""

from collections.abc import Iterable

from langgraph.graph import END, START, StateGraph

from agent_swarms.services.llm_provider import ChatProvider
from agent_swarms.services.spawn_policy import build_review_plan
from agent_swarms.state.enums import RunMode
from agent_swarms.state.schemas import GraphState

REVIEW_STRONG_HINTS = (
    "review",
    "audit",
    "production-ready",
    "repo",
    "repository",
    "codebase",
)
REVIEW_ACTION_HINTS = (
    "check",
    "analyze",
    "inspect",
    "scan",
    "run",
    "test",
    "build",
    "validate",
    "review",
    "audit",
)
REVIEW_SCOPE_HINTS = (
    "security",
    "secret",
    "vulnerability",
    "docs",
    "readme",
    "onboarding",
    "devex",
    "runtime",
    "test",
    "build",
    "dependency",
    "version",
    "frontend",
    "browser",
    "architecture",
    "maintainability",
    "refactor",
)


def _contains_review_hint(message: str) -> bool:
    if any(token in message for token in REVIEW_STRONG_HINTS):
        return True

    has_review_action = any(token in message for token in REVIEW_ACTION_HINTS)
    has_review_scope = any(token in message for token in REVIEW_SCOPE_HINTS)
    return has_review_action and has_review_scope


def _normalize_message_parts(parts: Iterable[str]) -> str:
    return " ".join(part for part in parts if part).strip()


def classify_request(state: GraphState) -> GraphState:
    """Infer whether the request is ordinary chat or review-oriented."""

    message = state["user_message"].strip().lower()
    requested_mode = state.get("requested_mode")
    is_review = requested_mode is RunMode.REVIEW or _contains_review_hint(message)
    if is_review:
        review_plan = build_review_plan(state["user_message"])
        route_reason = (
            "review mode was requested explicitly"
            if requested_mode is not None
            else "review intent detected"
        )
        route_reason_parts = [
            "supervisor classifier",
            route_reason,
            review_plan.route_reason,
        ]
        return {
            "mode": RunMode.REVIEW,
            "route_reason": _normalize_message_parts(route_reason_parts),
            "spawn_count": review_plan.spawn_count,
            "selected_workers": [worker.value for worker in review_plan.selected_workers],
            "needs_sandbox": review_plan.needs_sandbox,
        }

    route_reason_parts = [
        "supervisor classifier",
        "general chat detected",
    ]
    return {
        "mode": RunMode.CHAT,
        "route_reason": _normalize_message_parts(route_reason_parts),
        "spawn_count": 0,
        "selected_workers": [],
        "needs_sandbox": False,
    }


def build_main_graph(provider: ChatProvider):
    """Compile the supervisor routing workflow."""

    async def supervisor_chat(state: GraphState) -> GraphState:
        result = await provider.generate(
            user_message=state["user_message"],
            mode=state["mode"],
            route_reason=state["route_reason"],
        )
        return {
            "provider_backend": result.provider,
            "response": result.text,
        }

    workflow = StateGraph(GraphState)
    workflow.add_node("classify_request", classify_request)
    workflow.add_node("supervisor_chat", supervisor_chat)
    workflow.add_edge(START, "classify_request")
    workflow.add_edge("classify_request", "supervisor_chat")
    workflow.add_edge("supervisor_chat", END)
    return workflow.compile()
