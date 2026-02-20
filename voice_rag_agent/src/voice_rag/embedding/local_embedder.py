"""Embedding provider with NIM primary path and local fallback."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import TYPE_CHECKING

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from voice_rag.config import Settings

LOGGER = logging.getLogger("voice_rag.embedding")
WORD_PATTERN = re.compile(r"[a-z0-9]+")


class NimEmbeddingError(RuntimeError):
    """Raised when the NIM embedding request fails."""


class LocalEmbedder:
    """Generate embeddings via NIM with deterministic local fallback.

    Parameters
    ----------
    settings : Settings
        Application settings (provides NIM URL, API key, model name, etc.).
    """

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._dimension = settings.embedding_dimension
        self._url: str | None = None

        if settings.nim_base_url:
            self._url = (
                f"{str(settings.nim_base_url).rstrip('/')}"
                f"{settings.nim_embedding_path}"
            )
            LOGGER.info(
                "nim_embedder_init model=%s dim=%d url=%s",
                settings.nim_embedding_model,
                settings.embedding_dimension,
                self._url,
            )
        else:
            LOGGER.warning(
                "nim_embedder_disabled reason=missing_nim_base_url "
                "fallback=local_hash_embedding"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str, input_type: str = "query") -> list[float]:
        """Embed a single text string.

        Defaults to ``input_type="query"`` which is optimal for search
        queries.  Use ``"passage"`` when embedding document chunks.
        """
        normalized = text.strip()
        if not normalized:
            return [0.0] * self._dimension

        nim_vectors = self._embed_via_nim([normalized], input_type=input_type)
        if nim_vectors:
            return nim_vectors[0]
        return self._embed_locally(normalized)

    def embed_batch(
        self,
        texts: list[str],
        input_type: str = "passage",
    ) -> list[list[float]]:
        """Embed a batch of texts in one API call.

        Defaults to ``input_type="passage"`` which is optimal for
        document chunks during ingestion.
        """
        if not texts:
            return []

        cleaned = [t.strip() or " " for t in texts]
        nim_vectors = self._embed_via_nim(cleaned, input_type=input_type)
        if nim_vectors:
            return nim_vectors
        return [self._embed_locally(text) for text in cleaned]

    def _embed_via_nim(
        self,
        texts: list[str],
        input_type: str,
    ) -> list[list[float]] | None:
        """Try NIM embedding and return ``None`` when unavailable."""

        if not self._url:
            return None

        try:
            vectors = self._call_nim(texts, input_type=input_type)
        except NimEmbeddingError as error:
            LOGGER.warning(
                "nim_embedding_unavailable reason=%s fallback=local_hash_embedding",
                str(error),
            )
            return None

        if len(vectors) != len(texts):
            LOGGER.warning(
                "nim_embedding_length_mismatch expected=%d got=%d "
                "fallback=local_hash_embedding",
                len(texts),
                len(vectors),
            )
            return None

        return vectors

    # ------------------------------------------------------------------
    # NIM HTTP call with retry
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type((httpx.HTTPError, NimEmbeddingError)),
        reraise=True,
    )
    def _call_nim(
        self,
        texts: list[str],
        input_type: str,
    ) -> list[list[float]]:
        """POST to NIM ``/v1/embeddings`` and return embedding vectors."""
        if not self._url:
            raise NimEmbeddingError("NIM embedding URL is not configured.")

        headers: dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self._settings.nim_api_key:
            headers["Authorization"] = f"Bearer {self._settings.nim_api_key}"

        payload = {
            "model": self._settings.nim_embedding_model,
            "input": texts,
            "input_type": input_type,
            "encoding_format": "float",
            "truncate": "END",
        }

        try:
            with httpx.Client(
                timeout=self._settings.nim_timeout_seconds,
            ) as client:
                response = client.post(self._url, json=payload, headers=headers)
                response.raise_for_status()
        except httpx.HTTPError as error:
            LOGGER.warning("nim_embedding_request_failed err=%s", str(error))
            raise NimEmbeddingError("Embedding request to NIM failed.") from error

        data = response.json()
        return self._parse_response(data, expected_count=len(texts))

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        data: dict,
        expected_count: int,
    ) -> list[list[float]]:
        """Extract embedding vectors from NIM JSON response."""

        items = data.get("data")
        if not isinstance(items, list) or len(items) == 0:
            raise NimEmbeddingError("NIM embedding response has no data.")

        # Sort by index to guarantee order matches input
        items.sort(key=lambda item: item.get("index", 0))

        vectors: list[list[float]] = []
        for item in items:
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise NimEmbeddingError("NIM embedding item missing vector.")
            vectors.append(embedding)

        if len(vectors) != expected_count:
            LOGGER.warning(
                "nim_embedding_count_mismatch expected=%d got=%d",
                expected_count,
                len(vectors),
            )

        return vectors

    def _embed_locally(self, text: str) -> list[float]:
        """Build deterministic hash embeddings for offline resilience."""

        tokens = WORD_PATTERN.findall(text.lower())
        if not tokens:
            return [0.0] * self._dimension

        vector = [0.0] * self._dimension
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            first_index = (
                int.from_bytes(digest[:4], byteorder="little") % self._dimension
            )
            second_index = (
                int.from_bytes(digest[4:8], byteorder="little") % self._dimension
            )
            sign = 1.0 if digest[8] % 2 == 0 else -1.0

            vector[first_index] += sign
            vector[second_index] += sign * 0.5

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
