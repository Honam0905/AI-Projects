"""Agentic text RAG implemented with LangGraph."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from voice_rag.config import Settings
from voice_rag.embedding.local_embedder import LocalEmbedder
from voice_rag.llm.nim_chat import NimChatError, NimChatModel
from voice_rag.rag.hybrid_retriever import HybridRetriever
from voice_rag.rag.nim_rerank import NimRerankError, NimReranker
from voice_rag.rag.prompt import (
    build_context_text,
    build_system_prompt,
    build_user_prompt,
)
from voice_rag.vector_store.zvec_store import RetrievedChunk, ZvecStore

FALLBACK_ANSWER = "Not found in the provided documents."
LOGGER = logging.getLogger("voice_rag.rag.agent")


class RagState(TypedDict, total=False):
    """LangGraph runtime state."""

    kb_id: str
    question_text: str
    query_text: str
    query_embedding: list[float]
    retrieved_chunks: list[RetrievedChunk]
    rewrite_used: bool
    answer_text: str
    citations: list[dict[str, object]]


@dataclass(frozen=True)
class CitationResult:
    """Citation payload for chat response."""

    source_name: str
    doc_id: str
    page: int
    chunk_id: str
    snippet: str
    score: float
    bbox: list[float] | None


@dataclass(frozen=True)
class TextRagResult:
    """Text RAG output payload."""

    answer_text: str
    citations: list[CitationResult]


class TextRagAgent:
    """Text-only agentic RAG workflow."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embedder = LocalEmbedder(settings)
        self._zvec_store = ZvecStore(
            root_path=settings.data_dir / "zvec",
            collection_name=settings.zvec_collection_name,
            embedding_dimension=settings.embedding_dimension,
        )
        self._reranker: NimReranker | None = None
        if settings.rag_use_reranker and settings.nim_base_url:
            try:
                self._reranker = NimReranker(settings)
                LOGGER.info("nim_reranker_enabled model=%s", settings.nim_rerank_model)
            except NimRerankError as error:
                LOGGER.warning("nim_reranker_disabled reason=%s", str(error))
        self._hybrid_retriever = HybridRetriever(
            settings=settings,
            zvec_store=self._zvec_store,
            reranker=self._reranker,
        )
        self._llm_model: NimChatModel | None = None
        if settings.rag_use_llm and settings.nim_base_url:
            self._llm_model = NimChatModel(settings)
            LOGGER.info("nim_llm_enabled model=%s", settings.nim_llm_model)
        elif settings.rag_use_llm:
            LOGGER.warning(
                "nim_llm_disabled reason=missing_nim_base_url "
                "fallback=heuristic_generation"
            )

        self._compiled_graph = self._build_graph()

    def answer(self, kb_id: str, question_text: str) -> TextRagResult:
        """Run text RAG graph and return answer + citations."""

        graph_result = self._compiled_graph.invoke(
            {"kb_id": kb_id, "question_text": question_text}
        )

        raw_citations = graph_result.get("citations", [])
        citations = [
            CitationResult(
                source_name=str(citation.get("source_name", "")),
                doc_id=str(citation.get("doc_id", "")),
                page=int(citation.get("page", 0)),
                chunk_id=str(citation.get("chunk_id", "")),
                snippet=str(citation.get("snippet", "")),
                score=float(citation.get("score", 0.0)),
                bbox=None,
            )
            for citation in raw_citations
        ]
        answer_text = str(graph_result.get("answer_text", FALLBACK_ANSWER))

        return TextRagResult(answer_text=answer_text, citations=citations)

    def _build_graph(self):
        graph = StateGraph(RagState)

        graph.add_node("prepare_query", self._prepare_query)
        graph.add_node("embed_query", self._embed_query)
        graph.add_node("retrieve_chunks", self._retrieve_chunks)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("fallback_answer", self._fallback_answer)

        graph.add_edge(START, "prepare_query")
        graph.add_edge("prepare_query", "embed_query")
        graph.add_edge("embed_query", "retrieve_chunks")
        graph.add_conditional_edges(
            "retrieve_chunks",
            self._route_after_retrieval,
            {
                "rewrite_query": "rewrite_query",
                "generate_answer": "generate_answer",
                "fallback_answer": "fallback_answer",
            },
        )
        graph.add_edge("rewrite_query", "embed_query")
        graph.add_edge("generate_answer", END)
        graph.add_edge("fallback_answer", END)

        return graph.compile()

    def _prepare_query(self, state: RagState) -> RagState:
        question = str(state.get("question_text", "")).strip()
        return {
            "question_text": question,
            "query_text": question,
            "rewrite_used": False,
        }

    def _embed_query(self, state: RagState) -> RagState:
        query_text = str(state.get("query_text", "")).strip()
        query_embedding = self._embedder.embed(query_text)
        return {"query_embedding": query_embedding}

    def _retrieve_chunks(self, state: RagState) -> RagState:
        kb_id = str(state.get("kb_id", ""))
        question_text = str(state.get("query_text", ""))
        query_embedding = state.get("query_embedding", [])
        retrieved_chunks = self._hybrid_retriever.retrieve(
            kb_id=kb_id,
            question_text=question_text,
            query_embedding=query_embedding,
        )
        return {"retrieved_chunks": retrieved_chunks}

    def _rewrite_query(self, state: RagState) -> RagState:
        question_text = str(state.get("question_text", ""))
        keywords = _extract_keywords(question_text)
        if keywords:
            rewritten_query = " ".join(keywords)
        else:
            rewritten_query = question_text
        return {"query_text": rewritten_query, "rewrite_used": True}

    def _route_after_retrieval(self, state: RagState) -> str:
        retrieved_chunks = state.get("retrieved_chunks", [])
        if retrieved_chunks:
            return "generate_answer"
        if state.get("rewrite_used", False):
            return "fallback_answer"
        return "rewrite_query"

    def _generate_answer(self, state: RagState) -> RagState:
        question_text = str(state.get("question_text", ""))
        retrieved_chunks = state.get("retrieved_chunks", [])

        citations = self._build_citations(retrieved_chunks)
        answer_text = self._generate_answer_with_llm(question_text, retrieved_chunks)
        if not answer_text:
            answer_text = _build_grounded_answer(
                question_text=question_text,
                chunks=retrieved_chunks,
                max_answer_chars=self._settings.max_answer_chars,
            )

        return {"answer_text": answer_text, "citations": citations}

    def _build_citations(
        self,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[dict[str, object]]:
        citations: list[dict[str, object]] = []
        for chunk in retrieved_chunks[: self._settings.citation_top_k]:
            citations.append(
                {
                    "source_name": chunk.source_name,
                    "doc_id": chunk.doc_id,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "snippet": _shorten_text(chunk.text, max_chars=240),
                    "score": chunk.score,
                    "bbox": None,
                }
            )
        return citations

    def _generate_answer_with_llm(
        self,
        question_text: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> str:
        if not self._llm_model:
            return ""

        context_window = retrieved_chunks[: self._settings.retrieval_top_k]
        context_text = build_context_text(context_window)
        system_prompt = build_system_prompt(
            max_answer_chars=self._settings.max_answer_chars,
            fallback_answer=FALLBACK_ANSWER,
        )
        user_prompt = build_user_prompt(
            question_text=question_text,
            context_text=context_text,
        )

        try:
            raw_answer = self._llm_model.complete(system_prompt, user_prompt)
        except NimChatError as error:
            LOGGER.warning(
                "nim_llm_failed reason=%s fallback=heuristic_generation",
                str(error),
            )
            return ""

        cleaned_answer = _clean_answer(raw_answer)
        if not cleaned_answer:
            return ""
        return _shorten_text(cleaned_answer, max_chars=self._settings.max_answer_chars)

    def _fallback_answer(self, state: RagState) -> RagState:
        return {"answer_text": FALLBACK_ANSWER, "citations": []}


def _extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    stop_words = {
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
        "what",
        "which",
        "who",
        "where",
        "when",
        "why",
        "how",
        "tell",
        "about",
    }
    return [token for token in tokens if token not in stop_words][:8]


def _split_sentences(text: str) -> list[str]:
    segments = re.split(r"(?<=[.!?])\s+", text.strip())
    return [segment.strip() for segment in segments if segment.strip()]


def _shorten_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def _clean_answer(answer_text: str) -> str:
    cleaned = answer_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    return cleaned


def _build_grounded_answer(
    question_text: str,
    chunks: list[RetrievedChunk],
    max_answer_chars: int,
) -> str:
    if not chunks:
        return FALLBACK_ANSWER

    keywords = _extract_keywords(question_text)
    candidates: list[tuple[int, int, str]] = []

    for chunk in chunks:
        sentences = _split_sentences(chunk.text)
        if not sentences:
            sentences = [chunk.text]
        for sentence in sentences:
            lower_sentence = sentence.lower()
            keyword_hits = sum(1 for keyword in keywords if keyword in lower_sentence)
            candidates.append((keyword_hits, len(sentence), sentence))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    selected_sentences: list[str] = []
    total_chars = 0
    for keyword_hits, _, sentence in candidates:
        if keyword_hits == 0 and selected_sentences:
            continue
        if sentence in selected_sentences:
            continue

        separator_chars = 1 if selected_sentences else 0
        projected_length = total_chars + len(sentence) + separator_chars
        if projected_length > max_answer_chars:
            continue

        selected_sentences.append(sentence)
        total_chars = projected_length
        if len(selected_sentences) >= 3:
            break

    if not selected_sentences:
        return _shorten_text(chunks[0].text.strip(), max_chars=max_answer_chars)

    return " ".join(selected_sentences)
