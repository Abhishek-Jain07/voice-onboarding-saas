"""
Configuration management for the AI Voice Onboarding System.
Uses pydantic-settings for env-based configuration.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────
    app_name: str = "AI Voice Onboarding System"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # ── Server ───────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── CORS ─────────────────────────────────────────────────────────────
    cors_origins: str = "*"  # comma-separated origins

    # ── File limits ──────────────────────────────────────────────────────
    max_upload_size_mb: int = 50
    upload_dir: str = "uploads"

    # ── Model paths / flags ──────────────────────────────────────────────
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    spacy_model: str = "en_core_web_sm"

    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    zero_shot_model: str = "facebook/bart-large-mnli"

    # ── LLM defaults ────────────────────────────────────────────────────
    default_llm_provider: str = "openai"
    default_llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024

    # ── API keys (optional – can be passed per-request) ─────────────────
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"

    # ── Memory / profile storage ────────────────────────────────────────
    profile_storage_dir: str = "profiles"

    # ── Helpers ──────────────────────────────────────────────────────────
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def profile_path(self) -> Path:
        p = Path(self.profile_storage_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


# Singleton
settings = Settings()
