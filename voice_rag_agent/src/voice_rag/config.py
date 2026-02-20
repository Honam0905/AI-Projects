"""Application settings and helpers."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="VOICE_RAG_",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="voice-rag-agent")
    api_prefix: str = Field(default="/v1")
    env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    data_dir: Path = Field(default=Path("data"))
    nim_base_url: AnyHttpUrl | None = Field(default=None)
    nim_timeout_seconds: int = Field(default=30, ge=1, le=300)
    nim_retry_count: int = Field(default=2, ge=0, le=10)
    max_pdf_size_bytes: int = Field(default=25 * 1024 * 1024, ge=1)
    max_upload_files: int = Field(default=10, ge=1)
    chunk_size_chars: int = Field(default=1200, ge=50)
    chunk_overlap_chars: int = Field(default=150, ge=0)
    embedding_dimension: int = Field(default=1024, ge=8)
    nim_embedding_path: str = Field(default="/v1/embeddings")
    nim_embedding_model: str = Field(default="nvidia/nv-embedqa-e5-v5")
    zvec_collection_name: str = Field(default="kb_index_01")
    retrieval_top_k: int = Field(default=5, ge=1)
    retrieval_dense_top_k: int = Field(default=20, ge=1)
    retrieval_sparse_top_k: int = Field(default=20, ge=1)
    retrieval_hybrid_top_k: int = Field(default=20, ge=1)
    retrieval_rrf_k: int = Field(default=60, ge=1)
    citation_top_k: int = Field(default=3, ge=1)
    max_answer_chars: int = Field(default=500, ge=100)
    rag_use_llm: bool = Field(default=True)
    rag_use_reranker: bool = Field(default=True)
    voice_backend: Literal["local", "nim"] = Field(default="local")
    max_audio_size_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    nim_api_key: str | None = Field(default=None)
    nim_llm_path: str = Field(default="/v1/chat/completions")
    nim_llm_model: str = Field(default="nvidia/llama-3.3-nemotron-super-49b-v1.5")
    nim_llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    nim_llm_max_tokens: int = Field(default=256, ge=64, le=4096)
    nim_rerank_path: str = Field(default="/v1/ranking")
    nim_rerank_model: str = Field(default="nv-rerank-qa-mistral-4b:1")
    nim_rerank_truncate: str = Field(default="END")
    nim_asr_path: str = Field(default="/v1/audio/transcriptions")
    nim_tts_path: str = Field(default="/v1/audio/speech")
    nim_asr_model: str = Field(default="nvidia/parakeet-ctc-1.1b-asr")
    nim_tts_model: str = Field(default="nvidia/fastpitch-hifigan-tts")
    nim_tts_voice: str = Field(default="English-US.Female-1")
    nim_tts_format: str = Field(default="wav")


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()


def clear_settings_cache() -> None:
    """Clear settings cache for tests."""

    get_settings.cache_clear()
