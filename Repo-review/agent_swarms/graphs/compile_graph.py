"""Graph compilation helpers."""

from functools import lru_cache
from typing import Any

from agent_swarms.config import get_settings
from agent_swarms.graphs.main_graph import build_main_graph
from agent_swarms.services.llm_provider import get_chat_provider


@lru_cache
def get_compiled_graph() -> Any:
    """Build and cache the supervisor routing graph."""

    settings = get_settings()
    provider = get_chat_provider(settings)
    return build_main_graph(provider)
