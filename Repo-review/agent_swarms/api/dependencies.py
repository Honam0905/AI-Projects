"""Dependency helpers for API routes."""

from functools import lru_cache
from pathlib import Path

from agent_swarms.config import get_settings
from agent_swarms.sandbox.service import SandboxService
from agent_swarms.services.review_service import ReviewService
from agent_swarms.services.review_run_tracking import ReviewEventBroker, ReviewRunStore


@lru_cache
def get_review_run_store() -> ReviewRunStore:
    """Return a cached review run store."""

    settings = get_settings()
    return ReviewRunStore(Path(settings.sandbox_data_dir))


@lru_cache
def get_review_event_broker() -> ReviewEventBroker:
    """Return a cached review event broker."""

    return ReviewEventBroker(get_review_run_store())


@lru_cache
def get_sandbox_service() -> SandboxService:
    """Return a cached sandbox service."""

    return SandboxService(get_settings(), event_broker=get_review_event_broker())


@lru_cache
def get_review_service() -> ReviewService:
    """Return a cached review service."""

    settings = get_settings()
    return ReviewService(
        settings,
        get_sandbox_service(),
        get_review_run_store(),
        get_review_event_broker(),
    )
