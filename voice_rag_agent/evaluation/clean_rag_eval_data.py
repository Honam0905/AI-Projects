from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DATA_DIR = Path("/Users/honam/Cursor/voice_rag_agent/evaluation/rag_eval_data")
QA_PATH = DATA_DIR / "qa_gold.jsonl"
CHUNKS_PATH = DATA_DIR / "chunks_manifest.jsonl"
TTS_PATH = DATA_DIR / "tts_prompts.jsonl"

QUESTION_WORDS = {"what", "who", "when", "where", "which", "why", "how"}
ENTITY_JUNK_PATTERN = re.compile(r"##")
TOKEN_PATTERN = re.compile(r"\S+")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "\n".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows
    )
    path.write_text(payload + "\n", encoding="utf-8")


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def normalize_entity(text: str) -> str:
    text = normalize_space(text)
    text = text.strip("\"'`.,;:!?()[]{}")
    return normalize_space(text)


def should_drop_entity(entity: str) -> bool:
    if not entity:
        return True
    lowered = entity.lower()
    if lowered in QUESTION_WORDS:
        return True
    if ENTITY_JUNK_PATTERN.search(entity):
        return True

    alnum_chars = sum(ch.isalnum() for ch in entity)
    alpha_chars = sum(ch.isalpha() for ch in entity)
    if alnum_chars == 0:
        return True
    if alpha_chars == 0 and len(entity) < 4:
        return True

    return False


def recalc_length_bucket(word_count: int) -> str:
    return "long" if word_count >= 40 else "medium"


def clean_qa_rows(
    qa_rows: list[dict[str, Any]], chunk_map: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    cleaned_rows: list[dict[str, Any]] = []

    for row in qa_rows:
        ref_ids = row.get("reference_context_ids", [])
        context_text = " ".join(
            chunk_map[chunk_id]["text"]
            for chunk_id in ref_ids
            if chunk_id in chunk_map
            and isinstance(chunk_map[chunk_id].get("text"), str)
        )
        context_text_lower = context_text.lower()

        original_entities = row.get("reference_entities", [])
        if not isinstance(original_entities, list):
            original_entities = []

        cleaned_entities: list[str] = []
        seen_entities: set[str] = set()

        for raw_entity in original_entities:
            if not isinstance(raw_entity, str):
                continue
            entity = normalize_entity(raw_entity)
            if should_drop_entity(entity):
                continue

            key = entity.lower()
            if key in seen_entities:
                continue

            if context_text_lower and key not in context_text_lower:
                continue

            seen_entities.add(key)
            cleaned_entities.append(entity)

        new_row = dict(row)
        new_row["reference_entities"] = cleaned_entities
        cleaned_rows.append(new_row)

    return cleaned_rows


def clean_tts_rows(tts_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned_rows: list[dict[str, Any]] = []
    for row in tts_rows:
        text = row.get("text", "")
        if not isinstance(text, str):
            text = ""
        word_count = len(TOKEN_PATTERN.findall(text.strip()))

        new_row = dict(row)
        new_row["word_count"] = word_count
        new_row["length_bucket"] = recalc_length_bucket(word_count)
        cleaned_rows.append(new_row)

    return cleaned_rows


def main() -> None:
    qa_rows = read_jsonl(QA_PATH)
    chunk_rows = read_jsonl(CHUNKS_PATH)
    tts_rows = read_jsonl(TTS_PATH)

    chunk_map = {row["chunk_id"]: row for row in chunk_rows if "chunk_id" in row}

    cleaned_qa = clean_qa_rows(qa_rows, chunk_map)
    cleaned_tts = clean_tts_rows(tts_rows)

    write_jsonl(QA_PATH, cleaned_qa)
    write_jsonl(TTS_PATH, cleaned_tts)


if __name__ == "__main__":
    main()
