"""Review routes."""

from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)

from agent_swarms.api.dependencies import get_review_service
from agent_swarms.sandbox.base import SandboxConfigurationError, SandboxExecutionError
from agent_swarms.services.review_run_tracking import TERMINAL_RUN_STATUSES
from agent_swarms.services.review_service import ReviewService
from agent_swarms.state.schemas import (
    ReviewRequest,
    ReviewArtifactContentResponse,
    ReviewResponse,
    ReviewRunDetail,
    ReviewRunListResponse,
    ReviewRunSubmittedResponse,
)

router = APIRouter(tags=["review"])
SERVER_ERROR_EXCEPTIONS = (SandboxConfigurationError, SandboxExecutionError)


def _raise_http_error(exc: Exception) -> None:
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    if isinstance(exc, SERVER_ERROR_EXCEPTIONS):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected review error",
    ) from exc


@router.post("/review", response_model=ReviewResponse, status_code=status.HTTP_200_OK)
async def review_repo(
    request: ReviewRequest,
    service: ReviewService = Depends(get_review_service),
) -> ReviewResponse:
    """Run the repository review flow synchronously."""

    try:
        return await service.review(request)
    except Exception as exc:  # pragma: no cover - mapped by type
        _raise_http_error(exc)


@router.post(
    "/review/runs",
    response_model=ReviewRunSubmittedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_review_run(
    request: ReviewRequest,
    service: ReviewService = Depends(get_review_service),
) -> ReviewRunSubmittedResponse:
    """Submit an async repository review run."""

    try:
        return await service.submit_review(request)
    except Exception as exc:  # pragma: no cover - mapped by type
        _raise_http_error(exc)


@router.get("/review/runs", response_model=ReviewRunListResponse, status_code=status.HTTP_200_OK)
async def list_review_runs(
    service: ReviewService = Depends(get_review_service),
) -> ReviewRunListResponse:
    """List persisted review runs."""

    try:
        return service.list_runs()
    except Exception as exc:  # pragma: no cover - mapped by type
        _raise_http_error(exc)


@router.get("/review/runs/{run_id}", response_model=ReviewRunDetail, status_code=status.HTTP_200_OK)
async def get_review_run(
    run_id: str,
    service: ReviewService = Depends(get_review_service),
) -> ReviewRunDetail:
    """Return one persisted review run."""

    try:
        return service.get_run(run_id)
    except Exception as exc:  # pragma: no cover - mapped by type
        _raise_http_error(exc)


@router.get(
    "/review/runs/{run_id}/artifacts/content",
    response_model=ReviewArtifactContentResponse,
    status_code=status.HTTP_200_OK,
)
async def get_review_artifact_content(
    run_id: str,
    path: str,
    service: ReviewService = Depends(get_review_service),
) -> ReviewArtifactContentResponse:
    """Return the text content of one persisted review artifact."""

    try:
        return service.read_artifact_content(run_id, path)
    except Exception as exc:  # pragma: no cover - mapped by type
        _raise_http_error(exc)


@router.websocket("/review/runs/{run_id}/stream")
async def stream_review_run(
    websocket: WebSocket,
    run_id: str,
    service: ReviewService = Depends(get_review_service),
) -> None:
    """Replay and stream lifecycle events for one review run."""

    try:
        queue = await service.subscribe(run_id)
    except FileNotFoundError:
        await websocket.close(code=4404)
        return

    try:
        backlog = service.list_events(run_id)
        run = service.get_run(run_id)
    except Exception:
        await service.unsubscribe(run_id, queue)
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    await websocket.accept()
    seen_event_ids: set[str] = set()

    try:
        for event in backlog:
            if event.event_id in seen_event_ids:
                continue
            seen_event_ids.add(event.event_id)
            await websocket.send_json(event.model_dump(mode="json"))

        if run.status in TERMINAL_RUN_STATUSES:
            return

        while True:
            event = await queue.get()
            if event.event_id in seen_event_ids:
                continue
            seen_event_ids.add(event.event_id)
            await websocket.send_json(event.model_dump(mode="json"))
            if event.status in TERMINAL_RUN_STATUSES:
                return
    except WebSocketDisconnect:
        return
    finally:
        await service.unsubscribe(run_id, queue)
