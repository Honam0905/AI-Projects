"""NVIDIA NIM voice tools (ASR and TTS)."""

from __future__ import annotations

import base64
import binascii

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from voice_rag.config import Settings


class NimVoiceToolError(RuntimeError):
    """Raised when NIM voice endpoint call fails."""


class NimAsrTool:
    """ASR client using NIM HTTP endpoint."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._url = f"{str(settings.nim_base_url).rstrip('/')}{settings.nim_asr_path}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type((httpx.HTTPError, NimVoiceToolError)),
        reraise=True,
    )
    def transcribe(self, audio_bytes: bytes) -> str:
        headers = _nim_headers(self._settings)
        files = {
            "file": ("question.wav", audio_bytes, "audio/wav"),
            "model": (None, self._settings.nim_asr_model),
        }

        try:
            with httpx.Client(timeout=self._settings.nim_timeout_seconds) as client:
                response = client.post(self._url, files=files, headers=headers)
                response.raise_for_status()
        except httpx.HTTPError as error:
            raise NimVoiceToolError("ASR request to NIM failed.") from error

        payload = response.json()
        transcript = _extract_transcript(payload)
        if not transcript:
            raise NimVoiceToolError("ASR response did not contain transcript.")
        return transcript


class NimTtsTool:
    """TTS client using NIM HTTP endpoint."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._url = f"{str(settings.nim_base_url).rstrip('/')}{settings.nim_tts_path}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type((httpx.HTTPError, NimVoiceToolError)),
        reraise=True,
    )
    def synthesize(self, text: str) -> bytes:
        headers = _nim_headers(self._settings)
        payload = {
            "model": self._settings.nim_tts_model,
            "voice": self._settings.nim_tts_voice,
            "input": text,
            "response_format": self._settings.nim_tts_format,
        }

        try:
            with httpx.Client(timeout=self._settings.nim_timeout_seconds) as client:
                response = client.post(self._url, json=payload, headers=headers)
                response.raise_for_status()
        except httpx.HTTPError as error:
            raise NimVoiceToolError("TTS request to NIM failed.") from error

        content_type = response.headers.get("content-type", "")
        if content_type.startswith("audio/"):
            return response.content

        data = response.json()
        audio_b64 = _extract_audio_base64(data)
        if not audio_b64:
            raise NimVoiceToolError("TTS response did not contain audio data.")

        try:
            return base64.b64decode(audio_b64, validate=True)
        except (binascii.Error, ValueError) as error:
            raise NimVoiceToolError(
                "TTS response audio is not valid base64."
            ) from error


def _nim_headers(settings: Settings) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if settings.nim_api_key:
        headers["Authorization"] = f"Bearer {settings.nim_api_key}"
    return headers


def _extract_transcript(payload: dict) -> str:
    if isinstance(payload.get("text"), str):
        return payload["text"]
    if isinstance(payload.get("transcript"), str):
        return payload["transcript"]

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            if isinstance(first.get("text"), str):
                return first["text"]
            if isinstance(first.get("transcript"), str):
                return first["transcript"]
    return ""


def _extract_audio_base64(payload: dict) -> str:
    if isinstance(payload.get("audio_base64"), str):
        return payload["audio_base64"]
    if isinstance(payload.get("audio"), str):
        return payload["audio"]

    data = payload.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            if isinstance(first.get("b64_json"), str):
                return first["b64_json"]
            if isinstance(first.get("audio_base64"), str):
                return first["audio_base64"]
    return ""
