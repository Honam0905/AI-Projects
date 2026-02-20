"""NVIDIA NIM chat-completions client."""

from __future__ import annotations

import time
from typing import Any

import httpx

from voice_rag.config import Settings


class NimChatError(RuntimeError):
    """Raised when NIM chat request fails."""


class NimChatModel:
    """Simple NIM chat completions wrapper."""

    def __init__(self, settings: Settings) -> None:
        if not settings.nim_base_url:
            raise NimChatError("NIM base URL is required for LLM generation.")
        self._settings = settings
        self._url = f"{str(settings.nim_base_url).rstrip('/')}{settings.nim_llm_path}"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Generate one grounded completion string from NIM."""

        payload: dict[str, Any] = {
            "model": self._settings.nim_llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._settings.nim_llm_temperature,
        }
        payload["max_tokens"] = self._settings.nim_llm_max_tokens

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self._settings.nim_api_key:
            headers["Authorization"] = f"Bearer {self._settings.nim_api_key}"

        attempts = max(1, self._settings.nim_retry_count + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                with httpx.Client(timeout=self._settings.nim_timeout_seconds) as client:
                    response = client.post(self._url, json=payload, headers=headers)
                    response.raise_for_status()

                data = response.json()
                content = _extract_content_text(data).strip()
                if not content:
                    raise NimChatError("NIM LLM response did not contain answer text.")
                return content
            except (httpx.HTTPError, ValueError, NimChatError) as error:
                last_error = error
                if attempt == attempts - 1:
                    break
                delay_seconds = min(4.0, 0.5 * (2**attempt))
                time.sleep(delay_seconds)

        raise NimChatError("LLM request to NIM failed.") from last_error


def _extract_content_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""

    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
        return "".join(parts)

    return ""
