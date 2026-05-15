"""FastAPI application entrypoint."""

from fastapi import FastAPI

from agent_swarms import __version__
from agent_swarms.api.routes_chat import router as chat_router
from agent_swarms.api.routes_review import router as review_router
from agent_swarms.api.routes_sandbox import router as sandbox_router
from agent_swarms.config import get_settings
from agent_swarms.observability.logging import configure_logging


def create_app() -> FastAPI:
    """Create the FastAPI application."""

    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
    )

    @app.get("/health", tags=["system"])
    async def health_check() -> dict[str, str]:
        """Return basic runtime health for local development and demos."""

        return {
            "status": "ok",
            "environment": settings.app_env,
            "llm_backend": settings.llm_backend.value,
        }

    app.include_router(chat_router, prefix=settings.api_prefix)
    app.include_router(review_router, prefix=settings.api_prefix)
    app.include_router(sandbox_router, prefix=settings.api_prefix)
    return app


app = create_app()
