"""Prompt templates optimized for Ragas Context Precision and Answer Relevancy."""

from __future__ import annotations
from voice_rag.vector_store.zvec_store import RetrievedChunk

# ──────────────────────────────────────────────────────────────────────
# 1. QUERY REWRITING — Optimizing for Context Precision
# ──────────────────────────────────────────────────────────────────────


def build_rewrite_system_prompt() -> str:
    """Return system instructions for query rewriting (vector-optimized)."""
    return (
        "You are an expert search query generator for a semantic vector database.\n\n"
        "YOUR GOAL: Maximize retrieval precision by extracting the core semantic "
        "intent of the user's question and formatting it as a dense search query.\n\n"
        "RULES:\n"
        "1) Output exactly ONE line — the rewritten query. No quotes, no preamble.\n"
        "2) Isolate and preserve all core entities, domain-specific terminology, "
        "names, and dates.\n"
        "3) DO NOT add generic filler words (e.g., do not add 'definition', 'steps', "
        "'how-to', 'pros and cons'). These dilute the vector space.\n"
        "4) Deconstruct complex questions into a concise, declarative search string "
        "that represents the ideal document snippet answering the question.\n"
        "5) Resolve any ambiguous pronouns or abbreviations if the context is obvious.\n"
        "6) Keep the query focused strictly on the subject matter, stripping all "
        "conversational filler (e.g., 'tell me about', 'what is').\n"
    )


def build_rewrite_user_prompt(question_text: str) -> str:
    """Return user prompt for query rewriting."""
    question = _normalize_whitespace(question_text)
    return (
        f"Original user question: {question}\n\n"
        "Generate the optimized semantic search query:"
    )


# ──────────────────────────────────────────────────────────────────────
# 2. ANSWER GENERATION — Optimizing for Answer Relevancy
# ──────────────────────────────────────────────────────────────────────


def build_system_prompt(max_answer_chars: int, fallback_answer: str) -> str:
    """Return system instructions optimized for answer relevancy and grounding."""
    return (
        "You are an expert Document Q&A Assistant focused on extreme accuracy "
        "and directness.\n\n"
        "<instruction>\n"
        "**1. Core Mandate**\n"
        "Answer the user's question directly and comprehensively using ONLY the "
        "provided context chunks. Do not use external knowledge.\n\n"
        "**2. Driving Answer Relevancy**\n"
        "- BE DIRECT: Do not artificially repeat or mirror the user's question in "
        "your opening sentence. Start delivering the answer immediately.\n"
        "- BE COMPREHENSIVE: If the question has multiple parts, ensure every part "
        "is addressed based on the context.\n"
        "- NO FLUFF: Eliminate all preamble, redundant phrasing, filler, and "
        "conversational transitions. Every word must convey information.\n\n"
        "**3. Grounding and Citations**\n"
        "- Cite your sources strictly using chunk numbers inline (e.g., [1], [2]).\n"
        "- Place citations immediately after the specific claim they support, not "
        "just at the end of the paragraph.\n"
        "- If the provided context does not contain sufficient information to "
        "fully answer the question, respond with EXACTLY this phrase and nothing else:\n"
        f'"{fallback_answer}"\n\n'
        "**4. General Routing**\n"
        "- For basic greetings or arithmetic (e.g., 'Hello', '1+1'), answer "
        "directly without referencing documents or using citations.\n\n"
        "**5. Formatting Constraints**\n"
        "- Keep the answer concise. Aim for the highest information density possible.\n"
        "- Use bullet points ONLY if the question explicitly asks for a list or if "
        "the context provides a highly structured list.\n"
        f"- HARD LIMIT: {max_answer_chars} characters maximum.\n"
        "</instruction>\n"
    )


def build_user_prompt(question_text: str, context_text: str) -> str:
    """Return user prompt containing the question and retrieval context."""
    question = _normalize_whitespace(question_text)
    return (
        f"QUESTION: {question}\n\n"
        f"CONTEXT CHUNKS:\n{context_text}\n\n"
        "INSTRUCTION: Answer the question directly using only the context above. "
        "Do not repeat the question. Cite sources as [1], [2]. "
        f"If the answer is missing from the context, output the exact fallback sentence."
    )


# ──────────────────────────────────────────────────────────────────────
# 3. CONTEXT FORMATTING (Unchanged)
# ──────────────────────────────────────────────────────────────────────


def build_context_text(chunks: list[RetrievedChunk], max_chunk_chars: int = 800) -> str:
    """Format retrieval chunks as compact, unambiguous context blocks."""
    if not chunks:
        return "[No context chunks available]"

    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_text = _shorten_text(_normalize_whitespace(chunk.text), max_chunk_chars)
        lines.append(
            f"[{index}] (source: {chunk.source_name} | page: {chunk.page})\n"
            f"{chunk_text}"
        )
    return "\n\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# 4. UTILITIES (Unchanged)
# ──────────────────────────────────────────────────────────────────────


def _normalize_whitespace(text: str) -> str:
    """Collapse whitespace to reduce token noise and improve matching."""
    return " ".join(text.split()).strip()


def _shorten_text(text: str, max_chars: int) -> str:
    """Shorten text to a safe limit without breaking mid-word."""
    if len(text) <= max_chars:
        return text
    truncated = text[: max_chars - 3]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.7:
        truncated = truncated[:last_space]
    return f"{truncated}..."
