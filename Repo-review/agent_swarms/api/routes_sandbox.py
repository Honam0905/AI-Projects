"""Sandbox management routes."""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from agent_swarms.api.dependencies import get_sandbox_service
from agent_swarms.sandbox.base import SandboxConfigurationError, SandboxExecutionError
from agent_swarms.sandbox.service import SandboxService
from agent_swarms.state.schemas import (
    PythonExecutionRequest,
    SandboxArtifactsResponse,
    SandboxExecutionResponse,
    SandboxFileContentResponse,
    SandboxFileListResponse,
    SandboxRunCreateRequest,
    SandboxRunResponse,
    ShellExecutionRequest,
)

router = APIRouter(prefix="/sandbox", tags=["sandbox"])
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
        detail="Unexpected sandbox error",
    ) from exc


@router.post("/runs", response_model=SandboxRunResponse, status_code=status.HTTP_201_CREATED)
async def create_sandbox_run(
    request: SandboxRunCreateRequest,
    service: SandboxService = Depends(get_sandbox_service),
) -> SandboxRunResponse:
    """Create a sandbox-backed run."""

    try:
        return await service.create_run(request)
    except Exception as exc:  # pragma: no cover - mapped by type
        _raise_http_error(exc)


@router.get("/runs/{run_id}", response_model=SandboxRunResponse)
async def get_sandbox_run(
    run_id: str,
    service: SandboxService = Depends(get_sandbox_service),
) -> SandboxRunResponse:
    """Fetch sandbox run metadata."""

    try:
        return service.get_run(run_id)
    except Exception as exc:  # pragma: no cover - mapped by type
        _raise_http_error(exc)


@router.post("/runs/{run_id}/shell", response_model=SandboxExecutionResponse)
async def execute_shell(
    run_id: str,
    request: ShellExecutionRequest,
    service: SandboxService = Depends(get_sandbox_service),
) -> SandboxExecutionResponse:
    """Run a shell command inside the sandbox."""

    try:
        return await service.execute_shell(run_id, request)
    except Exception as exc:  # pragma: no cover - mapped by type
        _raise_http_error(exc)


@router.post("/runs/{run_id}/python", response_model=SandboxExecutionResponse)
async def execute_python(
    run_id: str,
    request: PythonExecutionRequest,
    service: SandboxService = Depends(get_sandbox_service),
) -> SandboxExecutionResponse:
    """Run Python code inside the sandbox."""

    try:
        return await service.execute_python(run_id, request)
    except Exception as exc:  # pragma: no cover - mapped by type
        _raise_http_error(exc)


@router.get("/runs/{run_id}/files", response_model=SandboxFileListResponse)
async def list_files(
    run_id: str,
    path: str = Query(default="."),
    service: SandboxService = Depends(get_sandbox_service),
) -> SandboxFileListResponse:
    """List files inside the sandbox workspace."""

    try:
        return await service.list_files(run_id, path)
    except Exception as exc:  # pragma: no cover - mapped by type
        _raise_http_error(exc)


@router.get("/runs/{run_id}/files/content", response_model=SandboxFileContentResponse)
async def read_file(
    run_id: str,
    path: str = Query(..., min_length=1),
    service: SandboxService = Depends(get_sandbox_service),
) -> SandboxFileContentResponse:
    """Read a file inside the sandbox workspace."""

    try:
        return await service.read_file(run_id, path)
    except Exception as exc:  # pragma: no cover - mapped by type
        _raise_http_error(exc)


@router.get("/runs/{run_id}/artifacts", response_model=SandboxArtifactsResponse)
async def list_artifacts(
    run_id: str,
    service: SandboxService = Depends(get_sandbox_service),
) -> SandboxArtifactsResponse:
    """List persisted artifacts for a sandbox run."""

    try:
        return service.list_artifacts(run_id)
    except Exception as exc:  # pragma: no cover - mapped by type
        _raise_http_error(exc)


@router.delete("/runs/{run_id}", response_model=SandboxRunResponse)
async def close_sandbox_run(
    run_id: str,
    service: SandboxService = Depends(get_sandbox_service),
) -> SandboxRunResponse:
    """Close a sandbox run."""

    try:
        return await service.close_run(run_id)
    except Exception as exc:  # pragma: no cover - mapped by type
        _raise_http_error(exc)
