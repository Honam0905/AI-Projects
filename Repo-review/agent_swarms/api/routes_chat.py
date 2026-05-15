"""Chat routes."""

from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from agent_swarms.graphs.compile_graph import get_compiled_graph
from agent_swarms.services.llm_provider import ProviderConfigurationError
from agent_swarms.state.enums import RunStatus
from agent_swarms.state.schemas import ChatRequest, ChatResponse, RunPlan

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(request: ChatRequest) -> ChatResponse:
    """Run the supervisor-only chat flow."""

    request_id = str(uuid4())
    graph = get_compiled_graph()

    try:
        result = await graph.ainvoke(
            {
                "request_id": request_id,
                "session_id": request.session_id,
                "user_message": request.message.strip(),
                "requested_mode": request.requested_mode,
            }
        )
    except ProviderConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return ChatResponse(
        request_id=request_id,
        status=RunStatus.COMPLETED,
        provider=result["provider_backend"],
        plan=RunPlan(
            mode=result["mode"],
            spawn_count=result["spawn_count"],
            selected_workers=result["selected_workers"],
            needs_sandbox=result["needs_sandbox"],
            route_reason=result["route_reason"],
        ),
        response=result["response"],
    )
