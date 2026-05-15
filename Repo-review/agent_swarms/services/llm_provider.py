"""Provider abstraction for supervisor chat."""

from dataclasses import dataclass
import re
from typing import Protocol

from agent_swarms.config import Settings
from agent_swarms.state.enums import ProviderBackend, RunMode

SUPERVISOR_SYSTEM_PROMPT = """
You are the main supervisor for an agent-swarms system.
Write like a practical product assistant, not an internal debug log.
Keep answers short, friendly, plain-language, and easy to scan.
Default to 2-4 sentences unless the user asks for deeper detail.
Use simple words such as "ready", "needs fixes", "changed files", and "next steps".
Avoid endpoint names, routing details, and implementation terms unless the user asks.
Do not claim that sub-agents or sandbox execution happened unless they actually did.
""".strip()


class ProviderConfigurationError(RuntimeError):
    """Raised when a configured provider cannot be initialized."""


@dataclass(frozen=True)
class ChatResult:
    """Normalized provider response."""

    text: str
    provider: ProviderBackend


class ChatProvider(Protocol):
    """Minimal protocol for the chat backend."""

    backend: ProviderBackend

    async def generate(
        self,
        *,
        user_message: str,
        mode: RunMode,
        route_reason: str,
    ) -> ChatResult:
        """Generate a normalized response."""


class MockChatProvider:
    """Safe local default used until a real provider is configured."""

    backend = ProviderBackend.MOCK

    async def generate(
        self,
        *,
        user_message: str,
        mode: RunMode,
        route_reason: str,
    ) -> ChatResult:
        """Generate a deterministic local supervisor response."""

        if mode is RunMode.REVIEW:
            text = (
                "This looks like a repository task, so it should run through the "
                "review swarm. Choose a repo target in the UI, then start the "
                "review so the workers can inspect it safely."
            )
        else:
            text = self._respond_to_simple_chat(user_message)

        return ChatResult(text=text, provider=self.backend)

    def _respond_to_simple_chat(self, user_message: str) -> str:
        message = user_message.strip()
        normalized = re.sub(r"\s+", " ", message.lower())

        if normalized in {"hi", "hello", "hey", "hi there", "hello there", "hey there"}:
            return (
                "Hi! I'm the supervisor agent for this Agent Swarms prototype. I "
                "can chat directly for simple questions, and I can route repo "
                "review requests into the swarm."
            )

        if "your name" in normalized or normalized in {"who are you", "what are you"}:
            return (
                "I'm the supervisor agent for Agent Swarms. "
                "I handle simple chat directly and dispatch review workers when "
                "a repo task needs the swarm."
            )

        if "which model" in normalized or "what model" in normalized:
            return (
                "Right now this response is coming from the mock supervisor backend. "
                "When the NVIDIA backend is enabled, the supervisor uses the "
                "configured ChatNVIDIA model."
            )

        return (
            "I'm the supervisor agent for this Agent Swarms prototype. "
            "For simple requests I answer directly here, and for repository work "
            "I start the review swarm."
        )


class NvidiaChatProvider:
    """Thin adapter around LangChain's ChatNVIDIA model."""

    backend = ProviderBackend.NVIDIA

    def __init__(self, settings: Settings) -> None:
        if not settings.nvidia_api_key:
            raise ProviderConfigurationError(
                "AGENT_SWARMS_NVIDIA_API_KEY is required when llm_backend=nvidia."
            )

        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise ProviderConfigurationError(
                "langchain-nvidia-ai-endpoints is not installed."
            ) from exc

        self._client = ChatNVIDIA(
            nvidia_api_key=settings.nvidia_api_key,
            model=settings.nvidia_model,
            temperature=0,
        )

    async def generate(
        self,
        *,
        user_message: str,
        mode: RunMode,
        route_reason: str,
    ) -> ChatResult:
        """Generate a supervisor response with ChatNVIDIA."""

        from langchain_core.messages import HumanMessage, SystemMessage

        response = await self._client.ainvoke(
            [
                SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Mode: {mode.value}\n"
                        f"Route reason: {route_reason}\n"
                        f"User message: {user_message}"
                    )
                ),
            ]
        )

        if isinstance(response.content, str):
            text = response.content.strip()
        else:  # pragma: no cover - depends on provider-specific content blocks
            text = " ".join(str(part) for part in response.content).strip()

        return ChatResult(text=text, provider=self.backend)


def get_chat_provider(settings: Settings) -> ChatProvider:
    """Return the configured provider implementation."""

    if settings.llm_backend is ProviderBackend.NVIDIA:
        return NvidiaChatProvider(settings)
    return MockChatProvider()
