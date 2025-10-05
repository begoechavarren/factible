from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Configuration
    api_v1_prefix: str = "/api/v1"
    project_name: str = "factible API"
    version: str = "0.1.0"
    debug: bool = False

    # CORS Configuration
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    # Factible Pipeline Configuration
    max_claims: int = 5
    max_queries_per_claim: int = 2
    max_results_per_query: int = 3
    headless_search: bool = True

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
