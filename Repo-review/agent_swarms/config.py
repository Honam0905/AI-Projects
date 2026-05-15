"""Application settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent_swarms.state.enums import ProviderBackend, SandboxBackend

DEFAULT_REVIEW_REPO_PATH = Path(__file__).resolve().parent.parent
DEFAULT_SANDBOX_DATA_DIR = DEFAULT_REVIEW_REPO_PATH.parent / ".agent_swarms_data"


class Settings(BaseSettings):
    """Typed environment-driven settings for the backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AGENT_SWARMS_",
        extra="ignore",
    )

    app_name: str = "Agent Swarms API"
    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)
    api_prefix: str = "/api"
    log_level: str = "INFO"
    llm_backend: ProviderBackend = ProviderBackend.MOCK
    nvidia_api_key: str | None = None
    nvidia_model: str = "meta/llama-3.1-70b-instruct"
    request_timeout_seconds: int = Field(default=45, ge=5, le=300)
    sandbox_backend: SandboxBackend = SandboxBackend.LOCAL
    sandbox_data_dir: str = str(DEFAULT_SANDBOX_DATA_DIR)
    sandbox_exec_timeout_seconds: int = Field(default=120, ge=5, le=900)
    sandbox_python_executable: str = "python3"
    docker_sandbox_image: str = "agent-swarms-sandbox:latest"
    docker_sandbox_container_prefix: str = "agent-swarms"
    default_review_repo_path: str = str(DEFAULT_REVIEW_REPO_PATH)
    repo_cache_dir: str | None = None
    repo_clone_timeout_seconds: int = Field(default=120, ge=10, le=900)
    local_clone_search_roots: str | None = None
    local_sandbox_use_macos_profile: bool = True
    open_sandbox_domain: str = "localhost:8080"
    open_sandbox_api_key: str | None = None
    open_sandbox_template: str = "ubuntu"
    open_sandbox_protocol: str = "http"


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
