"""Centralised, type-safe application configuration.

Every tunable knob referenced by the project brief (chunk size, overlap,
embedding model, LLM, temperature, top-k, database paths, upload size,
timeouts) is expressed here as a validated setting sourced from environment
variables. A single cached ``get_settings()`` accessor is the only supported
way to read configuration, which keeps the rest of the codebase free of
``os.environ`` look-ups and makes tests trivially overridable.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = three levels up from this file (app/shared/config.py -> root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AppEnv(StrEnum):
    """Deployment environment discriminator."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class RetrievalStrategy(StrEnum):
    """Supported retrieval algorithms."""

    SIMILARITY = "similarity"
    MMR = "mmr"


class LLMProvider(StrEnum):
    """Supported large-language-model providers.

    ``ECHO`` is a deterministic offline provider used for tests and for
    graceful degradation when no model backend is reachable.
    """

    GRANITE = "granite"
    LLAMA = "llama"
    OPENAI_COMPATIBLE = "openai_compatible"
    ECHO = "echo"


def _resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


class Settings(BaseSettings):
    """Application settings loaded from environment variables / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Application ----
    app_name: str = "AI Research Assistant"
    app_env: AppEnv = AppEnv.DEVELOPMENT
    app_debug: bool = True
    app_log_level: str = "INFO"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # ---- API / frontend wiring ----
    api_base_url: str = "http://localhost:8000"

    # ---- Storage ----
    storage_upload_dir: Path = Field(default=Path("storage/uploads"))
    storage_vectorstore_dir: Path = Field(default=Path("storage/vectorstore"))
    storage_database_path: Path = Field(default=Path("storage/database/metadata.db"))
    storage_log_dir: Path = Field(default=Path("storage/logs"))

    # ---- Upload limits ----
    upload_max_file_mb: int = Field(default=50, ge=1, le=1024)
    upload_allowed_mime: str = "application/pdf"

    # ---- Chunking ----
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)

    # ---- Embeddings ----
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=32, ge=1, le=512)

    # ---- Retrieval ----
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.MMR
    retrieval_mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    retrieval_score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    # ---- LLM ----
    llm_provider: LLMProvider = LLMProvider.GRANITE
    llm_model: str = "ibm-granite/granite-3.0-8b-instruct"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=1, le=32000)
    llm_timeout_seconds: int = Field(default=60, ge=1, le=600)
    llm_api_base: str = "http://localhost:11434/v1"
    llm_api_key: str = ""

    # ---- Authentication ----
    auth_enabled: bool = False
    auth_secret_key: str = "change-me-in-production-use-a-long-random-string"
    auth_token_ttl_minutes: int = Field(default=720, ge=1)

    # ---- Caching ----
    redis_enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"

    @field_validator(
        "storage_upload_dir",
        "storage_vectorstore_dir",
        "storage_database_path",
        "storage_log_dir",
        mode="after",
    )
    @classmethod
    def _resolve_paths(cls, value: Path) -> Path:
        """Ensure storage paths are absolute and rooted at the project."""
        return _resolve(value)

    @field_validator("chunk_overlap", mode="after")
    @classmethod
    def _overlap_less_than_size(cls, value: int, info) -> int:  # noqa: ANN001
        """Overlap must be strictly smaller than the chunk size."""
        size = info.data.get("chunk_size", 1000)
        if value >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value

    @property
    def upload_max_bytes(self) -> int:
        """Maximum upload size expressed in bytes."""
        return self.upload_max_file_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        """True when running in a production environment."""
        return self.app_env is AppEnv.PRODUCTION

    def ensure_directories(self) -> None:
        """Create all runtime storage directories if they do not exist."""
        for directory in (
            self.storage_upload_dir,
            self.storage_vectorstore_dir,
            self.storage_database_path.parent,
            self.storage_log_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide, cached settings instance."""
    return Settings()
