"""NVIDIA NIM reranker client."""

from __future__ import annotations

from urllib.parse import urlparse
import time
from dataclasses import dataclass
from typing import Any

import httpx

from voice_rag.config import Settings


class NimRerankError(RuntimeError):
    """Raised when NIM reranker call fails."""


@dataclass(frozen=True)
class RerankedItem:
    """Reranked item result."""

    index: int
    score: float


class NimReranker:
    """Simple HTTP client for NIM reranking endpoint."""

    def __init__(self, settings: Settings) -> None:
        if not settings.nim_base_url:
            raise NimRerankError("NIM base URL is required for reranking.")
        self._settings = settings
        self._base_url = str(settings.nim_base_url).rstrip("/")
        self._url = _join_api_url(self._base_url, settings.nim_rerank_path)

    def rerank(self, query: str, passages: list[str]) -> list[RerankedItem]:
        """Rerank candidate passages for a query."""

        if not passages:
            return []

        payload_candidates = self._payload_candidates(query=query, passages=passages)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self._settings.nim_api_key:
            headers["Authorization"] = f"Bearer {self._settings.nim_api_key}"

        attempts = max(1, self._settings.nim_retry_count + 1)
        candidate_urls = self._candidate_urls()
        last_error: Exception | None = None
        for attempt in range(attempts):
            for url in candidate_urls:
                for payload in payload_candidates:
                    try:
                        with httpx.Client(
                            timeout=self._settings.nim_timeout_seconds
                        ) as client:
                            response = client.post(url, json=payload, headers=headers)
                            response.raise_for_status()
                        return _parse_rerank_response(response.json())
                    except httpx.HTTPStatusError as error:
                        details = _extract_http_error_details(error.response)
                        last_error = NimRerankError(
                            f"Rerank HTTP {error.response.status_code} at {url}: {details}"
                        )
                    except (httpx.HTTPError, ValueError, NimRerankError) as error:
                        last_error = error

            if attempt == attempts - 1:
                break
            sleep_seconds = min(4.0, 0.5 * (2**attempt))
            time.sleep(sleep_seconds)

        raise NimRerankError(
            f"Rerank request to NIM failed: {last_error}"
        ) from last_error

    def _candidate_urls(self) -> list[str]:
        """Return candidate hosted/local endpoints for reranking."""

        candidates = [self._url]
        host = urlparse(self._base_url).netloc.lower()
        is_hosted = host in {"integrate.api.nvidia.com", "ai.api.nvidia.com"}
        if is_hosted:
            candidates.append("https://integrate.api.nvidia.com/v1/ranking")
            candidates.append("https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking")

        unique: list[str] = []
        seen: set[str] = set()
        for url in candidates:
            if url not in seen:
                seen.add(url)
                unique.append(url)
        return unique

    def _payload_candidates(
        self, query: str, passages: list[str]
    ) -> list[dict[str, Any]]:
        """Return payload variants for hosted and local compatibility."""

        base_payload: dict[str, Any] = {
            "model": self._settings.nim_rerank_model,
            "passages": [{"text": passage} for passage in passages],
        }
        if self._settings.nim_rerank_truncate:
            base_payload["truncate"] = self._settings.nim_rerank_truncate

        return [
            {**base_payload, "query": {"text": query}},
            {**base_payload, "query": query},
        ]


def _extract_http_error_details(response: httpx.Response) -> str:
    """Extract compact error details from response body."""

    content_type = response.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                if "error" in payload:
                    return str(payload["error"])
                if "message" in payload:
                    return str(payload["message"])
                return str(payload)
        except ValueError:
            pass
    text = response.text.strip()
    if not text:
        return "<empty body>"
    return text[:400]


def _join_api_url(base_url: str, path: str) -> str:
    """Join base URL and API path while preventing duplicated /v1."""

    normalized_path = path if path.startswith("/") else f"/{path}"
    if base_url.endswith("/v1") and normalized_path.startswith("/v1/"):
        return f"{base_url[:-3]}{normalized_path}"
    return f"{base_url}{normalized_path}"


def _parse_rerank_response(payload: dict[str, Any]) -> list[RerankedItem]:
    items = payload.get("rankings")
    if not isinstance(items, list):
        items = payload.get("data")
    if not isinstance(items, list):
        items = payload.get("results")
    if not isinstance(items, list):
        raise NimRerankError("Rerank response has no ranking list.")

    reranked: list[RerankedItem] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        score = item.get("logit")
        if score is None:
            score = item.get("relevance_score")
        if score is None:
            score = item.get("score")
        if not isinstance(index, int):
            continue
        if not isinstance(score, (int, float)):
            continue
        reranked.append(RerankedItem(index=index, score=float(score)))

    if not reranked:
        raise NimRerankError("Rerank response does not contain valid ranking items.")

    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked
