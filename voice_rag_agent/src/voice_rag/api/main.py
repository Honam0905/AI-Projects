"""FastAPI application entrypoint and middleware."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from voice_rag.api.routes import router
from voice_rag.api.schemas import ErrorDetail, ErrorResponse
from voice_rag.config import get_settings
from voice_rag.rag.agent import TextRagAgent
from voice_rag.voice.service import VoiceService

REQUEST_ID_HEADER = "X-Request-ID"
LOGGER = logging.getLogger("voice_rag.api")


def configure_logging(log_level: str) -> None:
    """Configure process logging."""

    level_name = log_level.upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def get_request_id(request: Request) -> str:
    """Read request id from state or create a new fallback id."""

    request_id = getattr(request.state, "request_id", "")
    if isinstance(request_id, str) and request_id:
        return request_id
    return str(uuid4())


def build_error_response(
    request_id: str,
    code: str,
    message: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    """Create error payload matching the API contract."""

    payload = ErrorResponse(
        error=ErrorDetail(code=code, message=message, details=details),
        request_id=request_id,
    )
    return payload.model_dump()


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach correlation id to request state, logs, and response headers."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid4())
        request.state.request_id = request_id

        started_at = perf_counter()
        response = await call_next(request)

        duration_ms = (perf_counter() - started_at) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        LOGGER.info(
            "request_completed request_id=%s method=%s path=%s status_code=%s "
            "duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


def register_exception_handlers(app: FastAPI) -> None:
    """Register exception handlers that return a standard error envelope."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        request_id = get_request_id(request)
        details = exc.detail if isinstance(exc.detail, dict) else {}
        message = exc.detail if isinstance(exc.detail, str) else "Request failed"
        return JSONResponse(
            status_code=exc.status_code,
            content=build_error_response(
                request_id=request_id,
                code=f"http_{exc.status_code}",
                message=message,
                details=details,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        request_id = get_request_id(request)
        return JSONResponse(
            status_code=400,
            content=build_error_response(
                request_id=request_id,
                code="validation_error",
                message="Request validation failed",
                details={"errors": exc.errors()},
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        request_id = get_request_id(request)
        LOGGER.exception("request_failed request_id=%s", request_id, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=build_error_response(
                request_id=request_id,
                code="internal_error",
                message="Internal server error",
                details={},
            ),
        )


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title=settings.app_name)
    app.state.settings = settings
    app.state.text_rag_agent = TextRagAgent(settings)
    app.state.voice_service = VoiceService(settings)
    app.add_middleware(RequestIdMiddleware)
    app.include_router(router)
    register_exception_handlers(app)
    return app
