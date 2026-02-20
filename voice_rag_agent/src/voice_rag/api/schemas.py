"""Shared API schema models."""

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Error payload details."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standard API error envelope."""

    error: ErrorDetail
    request_id: str


class CreateKbResponse(BaseModel):
    """Response for KB creation."""

    kb_id: str


class UploadedDocument(BaseModel):
    """Uploaded document metadata."""

    doc_id: str
    source_name: str
    pages: int


class UploadDocumentsResponse(BaseModel):
    """Response for document upload."""

    kb_id: str
    documents: list[UploadedDocument]


class Citation(BaseModel):
    """Grounding citation for an answer."""

    source_name: str
    doc_id: str
    page: int
    chunk_id: str
    snippet: str
    score: float
    bbox: list[float] | None = None


class ChatRequest(BaseModel):
    """Chat request payload."""

    mode: Literal["text", "voice"] = "text"
    question_text: str | None = None
    audio_base64: str | None = None
    read_aloud: bool = False


class ChatResponse(BaseModel):
    """Chat response payload."""

    answer_text: str
    citations: list[Citation]
    answer_audio_base64: str | None = None
