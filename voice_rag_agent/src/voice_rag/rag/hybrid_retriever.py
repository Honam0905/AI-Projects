"""Hybrid retrieval (vector + BM25) with optional NIM reranking."""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from voice_rag.config import Settings
from voice_rag.pdf.chunk import chunk_pages
from voice_rag.pdf.extract import PdfExtractionError, extract_page_texts
from voice_rag.rag.nim_rerank import NimRerankError, NimReranker
from voice_rag.vector_store.zvec_store import RetrievedChunk, ZvecStore

LOGGER = logging.getLogger("voice_rag.rag.hybrid")
WORD_PATTERN = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "and",
    "or",
    "this",
    "that",
    "it",
    "as",
    "at",
    "by",
    "from",
    "about",
    "what",
    "which",
    "who",
    "where",
    "when",
    "why",
    "how",
}


@dataclass(frozen=True)
class _SparseChunkDoc:
    chunk: RetrievedChunk
    token_counts: Counter[str]
    length: int


@dataclass(frozen=True)
class _SparseIndex:
    docs: dict[str, _SparseChunkDoc]
    doc_freq: Counter[str]
    average_length: float
    source_mtime_ns: int


@dataclass(frozen=True)
class _HybridCandidate:
    chunk: RetrievedChunk
    dense_rank: int | None
    sparse_rank: int | None
    dense_score: float
    sparse_score: float
    hybrid_score: float


class HybridRetriever:
    """Hybrid retrieval with reciprocal-rank fusion."""

    def __init__(
        self,
        settings: Settings,
        zvec_store: ZvecStore,
        reranker: NimReranker | None = None,
    ) -> None:
        self._settings = settings
        self._zvec_store = zvec_store
        self._reranker = reranker
        self._sparse_index_cache: dict[str, _SparseIndex] = {}

    def retrieve(
        self,
        kb_id: str,
        question_text: str,
        query_embedding: list[float],
    ) -> list[RetrievedChunk]:
        """Retrieve chunks using dense+sparse fusion and optional reranking."""

        dense_chunks = self._zvec_store.query_chunks(
            kb_id=kb_id,
            query_embedding=query_embedding,
            top_k=self._settings.retrieval_dense_top_k,
        )
        sparse_chunks = self._retrieve_sparse_chunks(
            kb_id=kb_id,
            question_text=question_text,
            top_k=self._settings.retrieval_sparse_top_k,
        )
        fused_candidates = self._fuse_rankings(dense_chunks, sparse_chunks)
        if not fused_candidates:
            return dense_chunks[: self._settings.retrieval_top_k]

        hybrid_top = fused_candidates[: self._settings.retrieval_hybrid_top_k]
        reranked = self._rerank(question_text=question_text, candidates=hybrid_top)
        if reranked:
            return reranked[: self._settings.retrieval_top_k]
        return [item.chunk for item in hybrid_top[: self._settings.retrieval_top_k]]

    def _retrieve_sparse_chunks(
        self,
        kb_id: str,
        question_text: str,
        top_k: int,
    ) -> list[_HybridCandidate]:
        sparse_index = self._load_sparse_index(kb_id=kb_id)
        if not sparse_index:
            return []

        query_tokens = _tokenize(question_text)
        if not query_tokens:
            return []

        total_docs = len(sparse_index.docs)
        scored: list[tuple[str, float]] = []
        for chunk_id, doc in sparse_index.docs.items():
            score = _bm25_score(
                query_tokens=query_tokens,
                doc=doc,
                total_docs=total_docs,
                doc_freq=sparse_index.doc_freq,
                avg_len=sparse_index.average_length,
            )
            if score <= 0:
                continue
            scored.append((chunk_id, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        top = scored[:top_k]
        return [
            _HybridCandidate(
                chunk=sparse_index.docs[chunk_id].chunk,
                dense_rank=None,
                sparse_rank=index + 1,
                dense_score=0.0,
                sparse_score=score,
                hybrid_score=0.0,
            )
            for index, (chunk_id, score) in enumerate(top)
        ]

    def _load_sparse_index(self, kb_id: str) -> _SparseIndex | None:
        chunk_path = self._settings.data_dir / "pdfs" / kb_id / "chunks.jsonl"
        if not chunk_path.exists():
            self._bootstrap_sparse_manifest(kb_id=kb_id, chunk_path=chunk_path)
            if not chunk_path.exists():
                return None

        source_mtime_ns = chunk_path.stat().st_mtime_ns
        cached = self._sparse_index_cache.get(kb_id)
        if cached and cached.source_mtime_ns == source_mtime_ns:
            return cached

        docs: dict[str, _SparseChunkDoc] = {}
        doc_freq: Counter[str] = Counter()
        total_length = 0

        for raw_line in chunk_path.read_text(encoding="utf-8").splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            row = json.loads(raw_line)
            text = str(row.get("text", ""))
            tokens = _tokenize(text)
            token_counts = Counter(tokens)
            for token in token_counts:
                doc_freq[token] += 1

            chunk = RetrievedChunk(
                chunk_id=str(row["chunk_id"]),
                doc_id=str(row.get("doc_id", "")),
                source_name=str(row.get("source_name", "")),
                page=int(row.get("page", 0)),
                chunk_index=int(row.get("chunk_index", 0)),
                text=text,
                score=0.0,
            )
            docs[chunk.chunk_id] = _SparseChunkDoc(
                chunk=chunk,
                token_counts=token_counts,
                length=max(1, sum(token_counts.values())),
            )
            total_length += docs[chunk.chunk_id].length

        if not docs:
            return None

        index = _SparseIndex(
            docs=docs,
            doc_freq=doc_freq,
            average_length=max(1.0, total_length / len(docs)),
            source_mtime_ns=source_mtime_ns,
        )
        self._sparse_index_cache[kb_id] = index
        return index

    def _bootstrap_sparse_manifest(self, kb_id: str, chunk_path: Path) -> None:
        """Build sparse chunk manifest from stored PDFs for older KBs."""

        kb_pdf_dir = self._settings.data_dir / "pdfs" / kb_id
        manifest_path = kb_pdf_dir / "manifest.json"
        if not manifest_path.exists():
            return

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        documents = manifest.get("documents", [])
        if not isinstance(documents, list) or not documents:
            return

        rows: list[dict[str, object]] = []
        for document in documents:
            if not isinstance(document, dict):
                continue
            doc_id = str(document.get("doc_id", "")).strip()
            if not doc_id:
                continue
            source_name = str(document.get("source_name", "")).strip()
            pdf_path = kb_pdf_dir / f"{doc_id}.pdf"
            if not pdf_path.exists():
                continue

            try:
                page_texts = extract_page_texts(pdf_path.read_bytes())
            except (OSError, PdfExtractionError):
                continue

            chunks = chunk_pages(
                pages=page_texts,
                chunk_size_chars=self._settings.chunk_size_chars,
                chunk_overlap_chars=self._settings.chunk_overlap_chars,
            )
            for chunk in chunks:
                rows.append(
                    {
                        "chunk_id": f"{doc_id}_{chunk.page}_{chunk.chunk_index}",
                        "doc_id": doc_id,
                        "source_name": source_name,
                        "page": chunk.page,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "created_at": 0,
                    }
                )

        if not rows:
            return

        with chunk_path.open("w", encoding="utf-8") as file_handle:
            for row in rows:
                file_handle.write(json.dumps(row, ensure_ascii=False))
                file_handle.write("\n")

    def _fuse_rankings(
        self,
        dense_chunks: list[RetrievedChunk],
        sparse_candidates: list[_HybridCandidate],
    ) -> list[_HybridCandidate]:
        dense_by_id = {chunk.chunk_id: chunk for chunk in dense_chunks}
        sparse_by_id = {
            candidate.chunk.chunk_id: candidate for candidate in sparse_candidates
        }
        candidate_ids = set(dense_by_id) | set(sparse_by_id)

        dense_rank_map = {
            chunk.chunk_id: rank for rank, chunk in enumerate(dense_chunks, start=1)
        }
        sparse_rank_map = {
            candidate.chunk.chunk_id: rank
            for rank, candidate in enumerate(sparse_candidates, start=1)
        }

        fused: list[_HybridCandidate] = []
        for chunk_id in candidate_ids:
            dense_chunk = dense_by_id.get(chunk_id)
            sparse_chunk = sparse_by_id.get(chunk_id)
            chunk = dense_chunk or (sparse_chunk.chunk if sparse_chunk else None)
            if not chunk:
                continue

            dense_rank = dense_rank_map.get(chunk_id)
            sparse_rank = sparse_rank_map.get(chunk_id)
            dense_score = float(dense_chunk.score) if dense_chunk else 0.0
            sparse_score = float(sparse_chunk.sparse_score) if sparse_chunk else 0.0
            hybrid_score = _rrf_score(
                dense_rank=dense_rank,
                sparse_rank=sparse_rank,
                rrf_k=self._settings.retrieval_rrf_k,
            )

            fused.append(
                _HybridCandidate(
                    chunk=chunk,
                    dense_rank=dense_rank,
                    sparse_rank=sparse_rank,
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                    hybrid_score=hybrid_score,
                )
            )

        fused.sort(key=lambda item: item.hybrid_score, reverse=True)
        return fused

    def _rerank(
        self,
        question_text: str,
        candidates: list[_HybridCandidate],
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        if not self._settings.rag_use_reranker or not self._reranker:
            return []

        passages = [candidate.chunk.text for candidate in candidates]
        try:
            reranked = self._reranker.rerank(query=question_text, passages=passages)
        except NimRerankError as error:
            LOGGER.warning("reranker_failed reason=%s fallback=hybrid_rank", str(error))
            return []

        ranked_chunks: list[RetrievedChunk] = []
        for item in reranked:
            if item.index < 0 or item.index >= len(candidates):
                continue
            base_chunk = candidates[item.index].chunk
            ranked_chunks.append(
                RetrievedChunk(
                    chunk_id=base_chunk.chunk_id,
                    doc_id=base_chunk.doc_id,
                    source_name=base_chunk.source_name,
                    page=base_chunk.page,
                    chunk_index=base_chunk.chunk_index,
                    text=base_chunk.text,
                    score=item.score,
                )
            )
        return ranked_chunks


def _tokenize(text: str) -> list[str]:
    tokens = WORD_PATTERN.findall(text.lower())
    return [token for token in tokens if token not in STOP_WORDS]


def _bm25_score(
    query_tokens: list[str],
    doc: _SparseChunkDoc,
    total_docs: int,
    doc_freq: Counter[str],
    avg_len: float,
) -> float:
    score = 0.0
    k1 = 1.5
    b = 0.75

    for token in query_tokens:
        tf = doc.token_counts.get(token, 0)
        if tf <= 0:
            continue
        freq = doc_freq.get(token, 0)
        idf = math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1.0)
        denominator = tf + k1 * (1.0 - b + b * doc.length / avg_len)
        score += idf * (tf * (k1 + 1.0)) / denominator
    return score


def _rrf_score(
    dense_rank: int | None,
    sparse_rank: int | None,
    rrf_k: int,
) -> float:
    score = 0.0
    if dense_rank is not None:
        score += 1.0 / (rrf_k + dense_rank)
    if sparse_rank is not None:
        score += 1.0 / (rrf_k + sparse_rank)
    return score
