"""API routes."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, WebSocket
from pydantic import ValidationError

from voice_rag.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    ErrorDetail,
    ErrorResponse,
    CreateKbResponse,
    UploadDocumentsResponse,
    UploadedDocument,
)
from voice_rag.kb.service import (
    InvalidDocumentError,
    KbNotFoundError,
    KbService,
    PayloadTooLargeError,
    UploadedDocumentInput,
)
from voice_rag.rag.agent import TextRagAgent
from voice_rag.vector_store.zvec_store import CollectionNotFoundError
from voice_rag.voice.service import VoiceInputError, VoiceService, VoiceToolError

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Return service liveness status."""

    return {"status": "ok"}


@router.post("/v1/kb", response_model=CreateKbResponse)
def create_kb(request: Request) -> CreateKbResponse:
    """Create a new knowledge base."""

    service = KbService(request.app.state.settings)
    kb_id = service.create_kb()
    return CreateKbResponse(kb_id=kb_id)


@router.post("/v1/kb/{kb_id}/documents", response_model=UploadDocumentsResponse)
async def upload_documents(
    kb_id: str,
    request: Request,
    files: list[UploadFile] = File(...),
) -> UploadDocumentsResponse:
    """Upload and ingest PDF documents into a KB."""

    service = KbService(request.app.state.settings)
    documents: list[UploadedDocumentInput] = []
    for uploaded_file in files:
        content = await uploaded_file.read()
        filename = uploaded_file.filename or "document.pdf"
        documents.append(
            UploadedDocumentInput(
                source_name=filename,
                content_type=uploaded_file.content_type,
                data=content,
            )
        )

    try:
        ingested_documents = service.ingest_documents(kb_id=kb_id, documents=documents)
    except KbNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except PayloadTooLargeError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    except InvalidDocumentError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return UploadDocumentsResponse(
        kb_id=kb_id,
        documents=[
            UploadedDocument(
                doc_id=document.doc_id,
                source_name=document.source_name,
                pages=document.pages,
            )
            for document in ingested_documents
        ],
    )


@router.post("/v1/kb/{kb_id}/chat", response_model=ChatResponse)
def chat(
    kb_id: str,
    payload: ChatRequest,
    request: Request,
) -> ChatResponse:
    """Run text or voice RAG chat for an existing KB."""

    try:
        return _execute_chat(payload=payload, kb_id=kb_id, app_state=request.app.state)
    except VoiceInputError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except VoiceToolError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except CollectionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.websocket("/v1/kb/{kb_id}/chat/ws")
async def chat_ws(kb_id: str, websocket: WebSocket) -> None:
    """Run streaming chat over WebSocket."""

    await websocket.accept()
    request_id = websocket.headers.get("X-Request-ID") or str(uuid4())

    try:
        raw_payload = await websocket.receive_json()
        payload = ChatRequest.model_validate(raw_payload)
    except ValidationError as error:
        await _send_ws_error(
            websocket=websocket,
            request_id=request_id,
            code="validation_error",
            message="Invalid chat payload.",
            status_code=400,
            details={"errors": error.errors()},
        )
        return
    except Exception as error:
        await _send_ws_error(
            websocket=websocket,
            request_id=request_id,
            code="invalid_json",
            message="Could not parse WebSocket payload.",
            status_code=400,
            details={"error": str(error)},
        )
        return

    await websocket.send_json(
        {
            "type": "progress",
            "request_id": request_id,
            "stage": "resolve_input",
        }
    )
    if payload.mode == "voice":
        await websocket.send_json(
            {"type": "progress", "request_id": request_id, "stage": "asr"}
        )
    await websocket.send_json(
        {"type": "progress", "request_id": request_id, "stage": "retrieve"}
    )

    try:
        chat_response = _execute_chat(
            payload=payload,
            kb_id=kb_id,
            app_state=websocket.scope["app"].state,
        )
    except VoiceInputError as error:
        await _send_ws_error(
            websocket=websocket,
            request_id=request_id,
            code="bad_request",
            message=str(error),
            status_code=400,
            details={},
        )
        return
    except CollectionNotFoundError as error:
        await _send_ws_error(
            websocket=websocket,
            request_id=request_id,
            code="not_found",
            message=str(error),
            status_code=404,
            details={},
        )
        return
    except VoiceToolError as error:
        await _send_ws_error(
            websocket=websocket,
            request_id=request_id,
            code="upstream_error",
            message=str(error),
            status_code=502,
            details={},
        )
        return
    except Exception as error:
        await _send_ws_error(
            websocket=websocket,
            request_id=request_id,
            code="internal_error",
            message="Internal server error",
            status_code=500,
            details={"error": str(error)},
        )
        return

    await websocket.send_json(
        {"type": "progress", "request_id": request_id, "stage": "generate"}
    )
    for token in _stream_tokens(chat_response.answer_text):
        await websocket.send_json(
            {
                "type": "token",
                "request_id": request_id,
                "text": token,
            }
        )

    if chat_response.answer_audio_base64:
        await websocket.send_json(
            {"type": "progress", "request_id": request_id, "stage": "tts"}
        )

    await websocket.send_json(
        {
            "type": "final",
            "request_id": request_id,
            "payload": chat_response.model_dump(),
        }
    )


def _execute_chat(payload: ChatRequest, kb_id: str, app_state) -> ChatResponse:
    """Execute chat request and return response payload."""

    text_rag_agent: TextRagAgent = app_state.text_rag_agent
    voice_service: VoiceService = app_state.voice_service

    voice_question = voice_service.resolve_question(
        mode=payload.mode,
        question_text=payload.question_text,
        audio_base64=payload.audio_base64,
    )

    rag_result = text_rag_agent.answer(
        kb_id=kb_id,
        question_text=voice_question.question_text,
    )

    answer_audio_base64: str | None = None
    should_speak = payload.mode == "voice" or payload.read_aloud
    if should_speak:
        answer_audio_base64 = voice_service.synthesize_answer_base64(
            rag_result.answer_text
        )

    return ChatResponse(
        answer_text=rag_result.answer_text,
        citations=[
            Citation(
                source_name=citation.source_name,
                doc_id=citation.doc_id,
                page=citation.page,
                chunk_id=citation.chunk_id,
                snippet=citation.snippet,
                score=citation.score,
                bbox=citation.bbox,
            )
            for citation in rag_result.citations
        ],
        answer_audio_base64=answer_audio_base64,
    )


def _stream_tokens(answer_text: str) -> list[str]:
    """Split answer into simple token chunks for streaming."""

    stripped = answer_text.strip()
    if not stripped:
        return []

    tokens = [f"{word} " for word in stripped.split()]
    if tokens:
        tokens[-1] = tokens[-1].rstrip()
    return tokens


async def _send_ws_error(
    websocket: WebSocket,
    request_id: str,
    code: str,
    message: str,
    status_code: int,
    details: dict[str, object],
) -> None:
    """Send structured WebSocket error event."""

    payload = ErrorResponse(
        error=ErrorDetail(code=code, message=message, details=details),
        request_id=request_id,
    )
    await websocket.send_json(
        {
            "type": "error",
            "request_id": request_id,
            "status_code": status_code,
            "payload": payload.model_dump(),
        }
    )
