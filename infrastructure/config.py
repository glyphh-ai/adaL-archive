"""
Ada configuration — simple, SQLite-only.

All configuration is read from environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Ada configuration."""

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8002, description="Server port")

    # Database — always SQLite
    database_url: Optional[str] = Field(
        default=None,
        description="SQLite database URL. Defaults to ./ada.db"
    )

    # Dev flags
    enable_docs: bool = Field(default=False, description="Enable /docs endpoint")
    cors_allow_all: bool = Field(default=True, description="Allow all CORS origins")

    # Storage — always SQLite
    storage_backend: str = Field(default="sqlite")

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        from pathlib import Path
        db_dir = Path.home() / ".ada"
        db_dir.mkdir(exist_ok=True)
        return f"sqlite+aiosqlite:///{db_dir}/ada.db"

    @property
    def resolved_storage_backend(self) -> str:
        return "sqlite"

    @property
    def resolved_max_vector_dimension(self) -> int:
        return 65536  # No practical limit for SQLite JSON storage

    # JWT (for token auth)
    jwt_secret_key: Optional[str] = Field(default=None)

    # Vector dimension
    max_vector_dimension: int = Field(default=65536)

    # Edge generation
    edge_generation_strategy: str = Field(default="lazy")
    edge_ttl_hours: int = Field(default=24)

    # Beam search
    default_beam_width: int = Field(default=5)
    default_max_tree_depth: int = Field(default=3)

    # Logging
    log_level: str = Field(default="INFO")

    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
    )
    cors_origins_production: List[str] = Field(default=[])

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def validate_settings() -> None:
    """Validate settings on startup."""
    get_settings()  # triggers pydantic validation
