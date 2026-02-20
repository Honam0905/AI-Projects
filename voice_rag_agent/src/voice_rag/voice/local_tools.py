"""Local ASR/TTS tool implementations for development and tests."""

from __future__ import annotations

import io
import json
import logging
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import httpx

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

LOGGER = logging.getLogger("voice_rag.voice.local_tools")


class LocalAsrTool:
    """Local ASR — sends raw PCM to Google Speech API (no FLAC binary needed)."""

    _GOOGLE_SPEECH_URL = "http://www.google.com/speech-api/v2/recognize"
    _GOOGLE_KEY = "AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw"  # public Chromium key

    def transcribe(self, audio_bytes: bytes) -> str:
        """Convert WAV audio bytes to text via Google Speech API."""

        if not audio_bytes:
            return ""

        # Extract raw PCM and sample rate from WAV
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                sample_rate = wf.getframerate()
                pcm_data = wf.readframes(wf.getnframes())
        except Exception as error:  # noqa: BLE001
            fallback_text = _decode_plain_text_audio(audio_bytes)
            if fallback_text:
                LOGGER.debug(
                    "local_asr_plain_text_fallback transcript=%.80s", fallback_text
                )
                return fallback_text
            LOGGER.warning("local_asr_wav_parse_failed err=%s", str(error))
            return ""

        if not pcm_data or len(pcm_data) < 100:
            LOGGER.warning("local_asr_pcm_too_short bytes=%d", len(pcm_data))
            return ""

        # Send raw PCM directly — no FLAC conversion needed
        params = {
            "client": "chromium",
            "lang": "en-US",
            "key": self._GOOGLE_KEY,
        }
        headers = {"Content-Type": f"audio/l16; rate={sample_rate}"}

        try:
            with httpx.Client(timeout=15) as client:
                response = client.post(
                    self._GOOGLE_SPEECH_URL,
                    params=params,
                    headers=headers,
                    content=pcm_data,
                )
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("local_asr_request_failed err=%s", str(error))
            return ""

        # Parse response (multiple JSON objects separated by newlines)
        transcript = ""
        for line in response.text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                for result in data.get("result", []):
                    alternatives = result.get("alternative", [])
                    if alternatives:
                        candidate = alternatives[0].get("transcript", "")
                        if candidate:
                            transcript = candidate
            except json.JSONDecodeError:
                continue

        if transcript:
            LOGGER.debug("local_asr_ok transcript=%.80s", transcript)
        else:
            LOGGER.warning("local_asr_empty_transcript response=%.200s", response.text)
        return transcript


class LocalTtsTool:
    """Local TTS with system voice fallback and gTTS support."""

    def synthesize(self, text: str) -> bytes:
        """Return playable audio bytes for local development.

        Tries in order: macOS ``say`` → gTTS (MP3) → silent WAV placeholder.
        """

        cleaned_text = _clean_text_for_speech(text)
        if not cleaned_text:
            return _build_silent_wav()

        speech_wav = _synthesize_with_macos_say(cleaned_text)
        if speech_wav:
            LOGGER.debug("tts_backend=macos_say bytes=%d", len(speech_wav))
            return speech_wav

        speech_mp3 = _synthesize_with_gtts(cleaned_text)
        if speech_mp3:
            LOGGER.debug("tts_backend=gtts bytes=%d", len(speech_mp3))
            return speech_mp3

        LOGGER.warning(
            "tts_all_backends_failed fallback=silent_wav "
            "hint='install gTTS (pip install gTTS) for reliable local TTS'"
        )
        return _build_silent_wav()


def _clean_text_for_speech(text: str) -> str:
    """Strip markdown formatting and artifacts that sound bad when spoken."""

    cleaned = text.strip()
    # Remove markdown headings
    cleaned = re.sub(r"#{1,6}\s*", "", cleaned)
    # Remove bold/italic markers
    cleaned = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", cleaned)
    cleaned = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", cleaned)
    # Remove inline code backticks
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    # Remove markdown links, keep text
    cleaned = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", cleaned)
    # Remove bullet point markers
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    # Remove numbered list markers
    cleaned = re.sub(r"^\s*\d+[.)]\s+", "", cleaned, flags=re.MULTILINE)
    # Collapse multiple newlines/spaces
    cleaned = re.sub(r"\n{2,}", ". ", cleaned)
    cleaned = re.sub(r"\n", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def _synthesize_with_macos_say(text: str) -> bytes:
    """Use macOS ``say`` to generate PCM WAV audio."""

    say_path = shutil.which("say")
    if not say_path:
        LOGGER.debug("tts_macos_say_not_found")
        return b""

    # Limit text length to avoid say timeout
    tts_text = text[:2000] if len(text) > 2000 else text

    with tempfile.TemporaryDirectory(prefix="voice_rag_tts_") as tmp_dir:
        output_path = Path(tmp_dir) / "speech.wav"
        command = [
            say_path,
            "-o",
            str(output_path),
            "--file-format=WAVE",
            "--data-format=LEI16@22050",
            tts_text,
        ]

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                LOGGER.warning(
                    "tts_macos_say_nonzero_exit code=%d stderr=%s",
                    result.returncode,
                    result.stderr.decode(errors="replace")[:200],
                )
                return b""
        except subprocess.CalledProcessError as error:
            LOGGER.warning(
                "tts_macos_say_failed code=%d stderr=%s",
                error.returncode,
                (error.stderr or b"").decode(errors="replace")[:200],
            )
            return b""
        except subprocess.TimeoutExpired:
            LOGGER.warning("tts_macos_say_timeout seconds=30")
            return b""
        except OSError as error:
            LOGGER.warning("tts_macos_say_oserror err=%s", str(error))
            return b""

        if not output_path.exists():
            LOGGER.warning("tts_macos_say_no_output")
            return b""

        try:
            wav_bytes = output_path.read_bytes()
        except OSError:
            return b""

    if wav_bytes.startswith(b"RIFF") and len(wav_bytes) > 44:
        return wav_bytes

    LOGGER.warning(
        "tts_macos_say_invalid_wav header=%s size=%d",
        wav_bytes[:4].hex() if wav_bytes else "empty",
        len(wav_bytes),
    )
    return b""


def _synthesize_with_gtts(text: str) -> bytes:
    """Use Google TTS (gTTS) to generate MP3 audio as fallback."""

    if gTTS is None:
        LOGGER.debug("tts_gtts_not_installed")
        return b""

    # Limit text for gTTS (max ~5000 chars)
    tts_text = text[:3000] if len(text) > 3000 else text

    try:
        tts = gTTS(text=tts_text, lang="en", slow=False)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        mp3_bytes = buffer.getvalue()
        if mp3_bytes and len(mp3_bytes) > 100:
            return mp3_bytes
        return b""
    except Exception as error:  # noqa: BLE001 — catch-all for optional dep
        LOGGER.warning("tts_gtts_failed err=%s", str(error))
        return b""


def _build_silent_wav() -> bytes:
    """Build a short silent WAV as last-resort placeholder."""

    sample_rate = 22_050
    duration_seconds = 0.1
    total_samples = int(sample_rate * duration_seconds)

    frames = b"\x00\x00" * total_samples

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)

    return buffer.getvalue()


def _decode_plain_text_audio(audio_bytes: bytes) -> str:
    """Decode plain text payload used by tests and dev stubs."""

    text = audio_bytes.decode("utf-8", errors="ignore").strip()
    if not text:
        return ""
    collapsed = " ".join(text.split())
    if not re.search(r"[a-zA-Z0-9]", collapsed):
        return ""
    return collapsed
