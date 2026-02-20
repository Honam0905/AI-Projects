"""Voice processing service for ASR and TTS."""

from __future__ import annotations

import base64
import binascii
import logging
import re
from dataclasses import dataclass
from typing import Protocol

from voice_rag.config import Settings
from voice_rag.voice.local_tools import LocalAsrTool, LocalTtsTool
from voice_rag.voice.nim_tools import NimAsrTool, NimTtsTool, NimVoiceToolError

LOGGER = logging.getLogger("voice_rag.voice.service")


class VoiceInputError(ValueError):
    """Raised when voice payload is invalid."""


class VoiceToolError(RuntimeError):
    """Raised when ASR or TTS tool execution fails."""


class AsrTool(Protocol):
    """ASR interface."""

    def transcribe(self, audio_bytes: bytes) -> str:
        """Convert speech audio into text."""


class TtsTool(Protocol):
    """TTS interface."""

    def synthesize(self, text: str) -> bytes:
        """Convert text into speech audio bytes."""


@dataclass(frozen=True)
class VoiceQuestion:
    """Normalized chat input after optional ASR."""

    question_text: str


class VoiceService:
    """Handle ASR input and TTS output orchestration."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._asr_tool, self._tts_tool = self._build_tools(settings)

    def resolve_question(
        self,
        mode: str,
        question_text: str | None,
        audio_base64: str | None,
    ) -> VoiceQuestion:
        """Resolve question text from text or voice payload."""

        normalized_mode = mode.strip().lower()
        if normalized_mode == "text":
            text = (question_text or "").strip()
            if not text:
                raise VoiceInputError("question_text is required for text mode.")
            return VoiceQuestion(question_text=text)

        if normalized_mode != "voice":
            raise VoiceInputError("Unsupported mode. Use 'text' or 'voice'.")

        if not audio_base64:
            raise VoiceInputError("audio_base64 is required for voice mode.")

        audio_bytes = _decode_audio_base64(audio_base64)
        if len(audio_bytes) > self._settings.max_audio_size_bytes:
            max_mb = self._settings.max_audio_size_bytes // (1024 * 1024)
            raise VoiceInputError(f"Audio payload exceeds size limit of {max_mb} MB.")

        try:
            transcript = self._asr_tool.transcribe(audio_bytes)
        except NimVoiceToolError as error:
            raise VoiceToolError(str(error)) from error

        transcript = transcript.strip()
        if not transcript:
            raise VoiceInputError("ASR produced an empty transcript.")

        return VoiceQuestion(question_text=transcript)

    def synthesize_answer_base64(self, answer_text: str) -> str:
        """Synthesize speech for answer and return base64 string."""

        cleaned_text = _clean_text_for_tts(answer_text)
        if not cleaned_text:
            raise VoiceInputError("answer_text is empty.")

        LOGGER.debug(
            "tts_input chars=%d text=%.80s...", len(cleaned_text), cleaned_text
        )

        try:
            audio_bytes = self._tts_tool.synthesize(cleaned_text)
        except NimVoiceToolError as error:
            raise VoiceToolError(str(error)) from error

        if not audio_bytes:
            raise VoiceToolError("TTS returned empty audio.")
        return base64.b64encode(audio_bytes).decode("utf-8")

    def _build_tools(self, settings: Settings) -> tuple[AsrTool, TtsTool]:
        if settings.voice_backend == "nim":
            if not settings.nim_base_url:
                raise VoiceToolError(
                    "VOICE_RAG_NIM_BASE_URL must be set when voice_backend is 'nim'."
                )
            LOGGER.info(
                "voice_backend=nim asr=%s tts=%s",
                settings.nim_asr_model,
                settings.nim_tts_model,
            )
            return NimAsrTool(settings), NimTtsTool(settings)

        # local mode: Google Speech API for ASR, gTTS/macOS say for TTS
        LOGGER.info("voice_backend=local asr=google_speech tts=gtts_or_say")
        return LocalAsrTool(), LocalTtsTool()


def _clean_text_for_tts(text: str) -> str:
    """Strip markdown formatting so TTS reads natural speech."""

    cleaned = text.strip()
    # Remove markdown headings
    cleaned = re.sub(r"#{1,6}\s*", "", cleaned)
    # Remove bold/italic markers
    cleaned = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", cleaned)
    cleaned = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", cleaned)
    # Remove inline code backticks
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    # Remove markdown links, keep display text
    cleaned = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", cleaned)
    # Remove bullet markers
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    # Remove numbered list markers
    cleaned = re.sub(r"^\s*\d+[.)]\s+", "", cleaned, flags=re.MULTILINE)
    # Convert multiple newlines to period-space for natural pausing
    cleaned = re.sub(r"\n{2,}", ". ", cleaned)
    cleaned = re.sub(r"\n", " ", cleaned)
    # Collapse whitespace
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def _decode_audio_base64(value: str) -> bytes:
    text = value.strip()
    if "," in text and text.lower().startswith("data:"):
        text = text.split(",", 1)[1]

    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as error:
        raise VoiceInputError("audio_base64 is not valid base64.") from error
