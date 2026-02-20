"""Zvec-backed vector store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from zvec import Collection, Doc, VectorQuery, create_and_open, open as zvec_open

from voice_rag.vector_store.zvec_schema import build_kb_collection_schema


class VectorStoreError(RuntimeError):
    """Base vector store error."""


class CollectionAlreadyExistsError(VectorStoreError):
    """Raised when trying to create an existing collection."""


class CollectionNotFoundError(VectorStoreError):
    """Raised when collection path does not exist."""


@dataclass(frozen=True)
class ChunkVector:
    """Chunk and vector data to upsert."""

    chunk_id: str
    doc_id: str
    source_name: str
    page: int
    chunk_index: int
    text: str
    embedding: list[float]
    created_at: int


@dataclass(frozen=True)
class RetrievedChunk:
    """Retrieved chunk record."""

    chunk_id: str
    doc_id: str
    source_name: str
    page: int
    chunk_index: int
    text: str
    score: float


class ZvecStore:
    """Simple wrapper for per-KB zvec collections."""

    def __init__(
        self,
        root_path: Path,
        collection_name: str,
        embedding_dimension: int,
    ) -> None:
        self._root_path = root_path
        self._collection_name = collection_name
        self._embedding_dimension = embedding_dimension

    def _collection_path(self, kb_id: str) -> Path:
        return self._root_path / kb_id

    def create_collection(self, kb_id: str) -> None:
        """Create a new KB collection."""

        collection_path = self._collection_path(kb_id)
        if collection_path.exists():
            raise CollectionAlreadyExistsError(
                f"Collection already exists for kb_id '{kb_id}'."
            )

        collection_path.parent.mkdir(parents=True, exist_ok=True)
        schema = build_kb_collection_schema(
            collection_name=self._collection_name,
            embedding_dimension=self._embedding_dimension,
        )
        create_and_open(path=str(collection_path), schema=schema)

    def open_collection(self, kb_id: str) -> Collection:
        """Open existing KB collection."""

        collection_path = self._collection_path(kb_id)
        if not collection_path.exists():
            raise CollectionNotFoundError(f"Collection not found for kb_id '{kb_id}'.")
        return zvec_open(path=str(collection_path))

    def upsert_chunks(self, kb_id: str, chunks: list[ChunkVector]) -> None:
        """Upsert chunk vectors into KB collection."""

        if not chunks:
            return

        collection = self.open_collection(kb_id)
        docs = [
            Doc(
                id=chunk.chunk_id,
                vectors={"embedding": chunk.embedding},
                fields={
                    "doc_id": chunk.doc_id,
                    "source_name": chunk.source_name,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "created_at": chunk.created_at,
                },
            )
            for chunk in chunks
        ]
        collection.upsert(docs)

    def query_chunks(
        self,
        kb_id: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Run vector similarity search for chunk retrieval."""

        collection = self.open_collection(kb_id)
        query = VectorQuery(field_name="embedding", vector=query_embedding)
        docs = collection.query(
            vectors=query,
            topk=top_k,
            output_fields=[
                "doc_id",
                "source_name",
                "page",
                "chunk_index",
                "text",
            ],
        )

        retrieved_chunks: list[RetrievedChunk] = []
        for doc in docs:
            fields = doc.fields or {}
            text = str(fields.get("text", ""))
            if not text:
                continue
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=str(doc.id),
                    doc_id=str(fields.get("doc_id", "")),
                    source_name=str(fields.get("source_name", "")),
                    page=int(fields.get("page", 0)),
                    chunk_index=int(fields.get("chunk_index", 0)),
                    text=text,
                    score=float(doc.score or 0.0),
                )
            )

        return retrieved_chunks
