"""Text chunking utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass

from voice_rag.pdf.extract import PageText

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class PageChunk:
    """Chunked text tied to a specific source page."""

    page: int
    chunk_index: int
    text: str


def chunk_pages(
    pages: list[PageText],
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> list[PageChunk]:
    """Split extracted pages into semantic sentence chunks with overlap."""

    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be greater than 0.")
    if chunk_overlap_chars < 0:
        raise ValueError("chunk_overlap_chars must be greater than or equal to 0.")
    if chunk_overlap_chars >= chunk_size_chars:
        raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars.")

    chunks: list[PageChunk] = []

    for page in pages:
        semantic_units = _extract_semantic_units(page.text)
        if not semantic_units:
            continue

        chunk_index = 0
        current_units: list[str] = []
        current_length = 0

        for unit in semantic_units:
            for fragment in _split_long_unit(unit, chunk_size_chars):
                fragment = fragment.strip()
                if not fragment:
                    continue

                separator = 1 if current_units else 0
                projected_length = current_length + len(fragment) + separator
                if current_units and projected_length > chunk_size_chars:
                    chunk_text = " ".join(current_units).strip()
                    if chunk_text:
                        chunks.append(
                            PageChunk(
                                page=page.page,
                                chunk_index=chunk_index,
                                text=chunk_text,
                            )
                        )
                        chunk_index += 1

                    current_units = _build_overlap_units(
                        current_units=current_units,
                        overlap_chars=chunk_overlap_chars,
                    )
                    current_length = len(" ".join(current_units))

                separator = 1 if current_units else 0
                projected_length = current_length + len(fragment) + separator
                if projected_length > chunk_size_chars:
                    if current_units:
                        chunk_text = " ".join(current_units).strip()
                        if chunk_text:
                            chunks.append(
                                PageChunk(
                                    page=page.page,
                                    chunk_index=chunk_index,
                                    text=chunk_text,
                                )
                            )
                            chunk_index += 1
                    current_units = [fragment]
                    current_length = len(fragment)
                else:
                    current_units.append(fragment)
                    current_length = projected_length

        if current_units:
            chunk_text = " ".join(current_units).strip()
            if chunk_text:
                chunks.append(
                    PageChunk(
                        page=page.page,
                        chunk_index=chunk_index,
                        text=chunk_text,
                    )
                )

    return chunks


def _extract_semantic_units(text: str) -> list[str]:
    """Extract sentence-like semantic units from raw page text."""

    normalized = text.strip()
    if not normalized:
        return []

    units: list[str] = []
    paragraphs = re.split(r"\n\s*\n", normalized)
    for paragraph in paragraphs:
        paragraph_text = " ".join(paragraph.split()).strip()
        if not paragraph_text:
            continue

        sentences = [
            sentence.strip()
            for sentence in SENTENCE_SPLIT_PATTERN.split(paragraph_text)
            if sentence.strip()
        ]
        if sentences:
            units.extend(sentences)
        else:
            units.append(paragraph_text)

    return units


def _split_long_unit(unit: str, chunk_size_chars: int) -> list[str]:
    """Split very long sentence/paragraph into smaller fragments."""

    if len(unit) <= chunk_size_chars:
        return [unit]

    words = unit.split()
    fragments: list[str] = []
    current_words: list[str] = []

    for word in words:
        candidate_words = current_words + [word]
        candidate = " ".join(candidate_words)
        if current_words and len(candidate) > chunk_size_chars:
            fragments.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words = candidate_words

    if current_words:
        fragments.append(" ".join(current_words))

    return [fragment for fragment in fragments if fragment.strip()]


def _build_overlap_units(current_units: list[str], overlap_chars: int) -> list[str]:
    """Keep trailing semantic units as overlap for next chunk."""

    if overlap_chars <= 0 or not current_units:
        return []

    overlap_units: list[str] = []
    total_chars = 0
    for unit in reversed(current_units):
        additional_chars = len(unit) + (1 if overlap_units else 0)
        if total_chars >= overlap_chars:
            break
        overlap_units.insert(0, unit)
        total_chars += additional_chars
    return overlap_units
