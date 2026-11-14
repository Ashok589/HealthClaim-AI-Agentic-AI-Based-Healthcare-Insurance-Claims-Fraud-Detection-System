"""
app/utils/config.py
Application configuration via Pydantic BaseSettings.
All values are loaded from environment variables / .env file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Google AI
    # ------------------------------------------------------------------ #
    google_api_key: str = Field(default="", description="Google Generative AI API key")
    llm_model: str = Field(default="gemini-pro", description="Gemini model name for LLM calls")
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model for embeddings",
    )

    # ------------------------------------------------------------------ #
    # Database
    # ------------------------------------------------------------------ #
    database_url: str = Field(
        default="sqlite:///./healthclaim.db",
        description="SQLAlchemy database connection string",
    )

    # ------------------------------------------------------------------ #
    # Security
    # ------------------------------------------------------------------ #
    secret_key: str = Field(
        default="change_me_in_production",
        description="JWT / session secret key",
    )
    algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    access_token_expire_minutes: int = Field(
        default=60, description="JWT token lifetime in minutes"
    )

    # ------------------------------------------------------------------ #
    # App behaviour
    # ------------------------------------------------------------------ #
    debug: bool = Field(default=True, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level (DEBUG/INFO/WARNING/ERROR)")
    app_name: str = Field(default="HealthClaim AI", description="Human-readable application name")
    app_version: str = Field(default="1.0.0", description="Application version string")

    # ------------------------------------------------------------------ #
    # ChromaDB (vector store)
    # ------------------------------------------------------------------ #
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        description="Directory where ChromaDB persists its data",
    )
    chroma_collection_name: str = Field(
        default="healthclaim_docs",
        description="ChromaDB collection name for policy/document embeddings",
    )

    # ------------------------------------------------------------------ #
    # Fraud detection
    # ------------------------------------------------------------------ #
    fraud_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Score above which a claim is flagged as suspected fraud",
    )

    # ------------------------------------------------------------------ #
    # File storage
    # ------------------------------------------------------------------ #
    upload_dir: str = Field(
        default="./data/claims",
        description="Directory where uploaded claim documents are stored",
    )
    max_upload_size_mb: int = Field(
        default=20,
        description="Maximum allowed file upload size in megabytes",
    )

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper

    @field_validator("fraud_threshold", mode="before")
    @classmethod
    def _validate_fraud_threshold(cls, v: float) -> float:
        v = float(v)
        if not 0.0 <= v <= 1.0:
            raise ValueError("fraud_threshold must be between 0.0 and 1.0")
        return v

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @property
    def upload_path(self) -> Path:
        """Return the upload directory as a resolved Path, creating it if needed."""
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chroma_path(self) -> Path:
        """Return the ChromaDB persist directory as a resolved Path."""
        p = Path(self.chroma_persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


# Module-level singleton — import this everywhere
settings = Settings()
