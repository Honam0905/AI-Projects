"""Knowledge base creation and document ingestion service."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from voice_rag.config import Settings
from voice_rag.embedding.local_embedder import LocalEmbedder
from voice_rag.pdf.chunk import chunk_pages
from voice_rag.pdf.extract import PdfExtractionError, extract_page_texts
from voice_rag.vector_store.zvec_store import ChunkVector, ZvecStore

PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}


class KbServiceError(RuntimeError):
    """Base knowledge base service error."""


class KbNotFoundError(KbServiceError):
    """Raised when kb_id does not exist."""


class InvalidDocumentError(KbServiceError):
    """Raised for invalid uploaded files."""


class PayloadTooLargeError(KbServiceError):
    """Raised when uploaded file exceeds configured limit."""


@dataclass(frozen=True)
class UploadedDocumentInput:
    """Uploaded file payload."""

    source_name: str
    content_type: str | None
    data: bytes


@dataclass(frozen=True)
class UploadedDocumentResult:
    """Ingestion result metadata."""

    doc_id: str
    source_name: str
    pages: int


class KbService:
    """Service for KB lifecycle and PDF ingestion."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pdf_root = settings.data_dir / "pdfs"
        self._zvec_store = ZvecStore(
            root_path=settings.data_dir / "zvec",
            collection_name=settings.zvec_collection_name,
            embedding_dimension=settings.embedding_dimension,
        )
        self._embedder = LocalEmbedder(settings)

    def create_kb(self) -> str:
        """Create a new knowledge base and backing storage."""

        kb_id = str(uuid4())
        kb_pdf_dir = self._pdf_root / kb_id
        kb_pdf_dir.mkdir(parents=True, exist_ok=False)

        try:
            self._zvec_store.create_collection(kb_id)
        except Exception as error:
            shutil.rmtree(kb_pdf_dir, ignore_errors=True)
            raise KbServiceError("Failed to create vector collection.") from error

        return kb_id

    def ingest_documents(
        self,
        kb_id: str,
        documents: list[UploadedDocumentInput],
    ) -> list[UploadedDocumentResult]:
        """Ingest PDF documents into an existing KB."""

        kb_pdf_dir = self._pdf_root / kb_id
        if not kb_pdf_dir.is_dir():
            raise KbNotFoundError(f"Knowledge base '{kb_id}' not found.")
        if not documents:
            raise InvalidDocumentError("No files were provided.")
        if len(documents) > self._settings.max_upload_files:
            raise InvalidDocumentError(
                f"Maximum {self._settings.max_upload_files} files allowed per upload."
            )

        ingested: list[UploadedDocumentResult] = []
        for document in documents:
            result = self._ingest_single_document(
                kb_id=kb_id,
                kb_pdf_dir=kb_pdf_dir,
                document=document,
            )
            ingested.append(result)

        self._update_manifest(kb_id=kb_id, documents=ingested)
        return ingested

    def _ingest_single_document(
        self,
        kb_id: str,
        kb_pdf_dir: Path,
        document: UploadedDocumentInput,
    ) -> UploadedDocumentResult:
        source_name = Path(document.source_name).name or "document.pdf"
        self._validate_document(source_name=source_name, document=document)

        try:
            pages = extract_page_texts(document.data)
        except PdfExtractionError as error:
            raise InvalidDocumentError(str(error)) from error

        chunks = chunk_pages(
            pages=pages,
            chunk_size_chars=self._settings.chunk_size_chars,
            chunk_overlap_chars=self._settings.chunk_overlap_chars,
        )
        if not chunks:
            raise InvalidDocumentError("PDF has no chunkable text.")

        doc_id = str(uuid4())
        pdf_path = kb_pdf_dir / f"{doc_id}.pdf"
        pdf_path.write_bytes(document.data)

        created_at = int(time.time() * 1000)

        # Batch embed all chunks at once for efficiency
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self._embedder.embed_batch(chunk_texts)

        chunk_vectors: list[ChunkVector] = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = f"{doc_id}_{chunk.page}_{chunk.chunk_index}"
            chunk_vectors.append(
                ChunkVector(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source_name=source_name,
                    page=chunk.page,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    embedding=embedding,
                    created_at=created_at,
                )
            )

        self._zvec_store.upsert_chunks(kb_id=kb_id, chunks=chunk_vectors)
        self._append_chunk_manifest(kb_pdf_dir=kb_pdf_dir, chunks=chunk_vectors)
        return UploadedDocumentResult(
            doc_id=doc_id, source_name=source_name, pages=len(pages)
        )

    def _validate_document(
        self,
        source_name: str,
        document: UploadedDocumentInput,
    ) -> None:
        if not document.data:
            raise InvalidDocumentError(f"File '{source_name}' is empty.")

        if len(document.data) > self._settings.max_pdf_size_bytes:
            max_size_mb = self._settings.max_pdf_size_bytes // (1024 * 1024)
            raise PayloadTooLargeError(
                f"File '{source_name}' exceeds size limit of {max_size_mb} MB."
            )

        content_type = (document.content_type or "").lower()
        is_pdf_name = source_name.lower().endswith(".pdf")
        is_pdf_type = content_type in PDF_CONTENT_TYPES
        if not is_pdf_name and not is_pdf_type:
            raise InvalidDocumentError(f"File '{source_name}' must be a PDF.")

    def _update_manifest(
        self,
        kb_id: str,
        documents: list[UploadedDocumentResult],
    ) -> None:
        kb_pdf_dir = self._pdf_root / kb_id
        manifest_path = kb_pdf_dir / "manifest.json"

        manifest = {"kb_id": kb_id, "documents": []}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        for document in documents:
            manifest["documents"].append(asdict(document))

        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    def _append_chunk_manifest(
        self,
        kb_pdf_dir: Path,
        chunks: list[ChunkVector],
    ) -> None:
        """Persist chunk metadata for sparse retrieval and debugging."""

        if not chunks:
            return

        chunk_manifest_path = kb_pdf_dir / "chunks.jsonl"
        with chunk_manifest_path.open("a", encoding="utf-8") as file_handle:
            for chunk in chunks:
                row = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "source_name": chunk.source_name,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "created_at": chunk.created_at,
                }
                file_handle.write(json.dumps(row, ensure_ascii=False))
                file_handle.write("\n")
