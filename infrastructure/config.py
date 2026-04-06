"""
Runtime configuration management using Pydantic settings.

All configuration is read from environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration settings"""

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8002, description="Server port")

    # Database — auto-defaults to SQLite if not set
    database_url: Optional[str] = Field(
        default=None,
        description="Database connection URL. Defaults to SQLite if not set."
    )

    # UI/Dev flags
    enable_docs: bool = Field(
        default=False,
        description="Enable /docs and /redoc OpenAPI endpoints"
    )
    cors_allow_all: bool = Field(
        default=False,
        description="Allow all CORS origins (set by CLI, not for production)"
    )

    # Storage backend
    storage_backend: Literal["auto", "pgvector", "sqlite", "memory"] = Field(
        default="auto",
        description=(
            "Storage backend for glyph persistence. "
            "'auto' detects from database_url (sqlite:// → sqlite, postgresql:// → pgvector). "
            "'pgvector' requires DATABASE_URL pointing to PostgreSQL + pgvector. "
            "'sqlite' uses SQLite with Python cosine similarity (dev/small datasets). "
            "'memory' reserved for future in-process backends (Pinecone, Qdrant, etc.)."
        ),
    )

    @property
    def resolved_database_url(self) -> str:
        """Return the effective database URL, defaulting to SQLite if not set."""
        if self.database_url:
            return self.database_url
        return "sqlite+aiosqlite:///./ada_dev.db"

    @property
    def resolved_storage_backend(self) -> str:
        """Return the effective storage backend, auto-detecting from URL if 'auto'."""
        if self.storage_backend != "auto":
            return self.storage_backend
        url = self.resolved_database_url
        if "sqlite" in url:
            return "sqlite"
        if "postgresql" in url or "postgres" in url:
            return "pgvector"
        return "sqlite"  # safe fallback
    
    # JWT Authentication — shared secret with Platform.
    # Database tokens are the primary auth for CLI/API.
    jwt_secret_key: Optional[str] = Field(
        default=None,
        description="Platform JWT secret (HS256). Set to same value as Platform's JWT_SECRET_KEY."
    )
    
    # NL Query (enabled by default for studio chat functionality)
    enable_nl_query: bool = Field(
        default=True,
        description="Enable natural language query interface"
    )
    
    # Rate limiting
    rate_limit_per_minute: int = Field(
        default=60,
        description="Default rate limit per minute"
    )
    
    # Resource quotas
    default_model_memory_mb: int = Field(
        default=1024,
        description="Default memory quota per model in MB"
    )
    default_model_storage_gb: int = Field(
        default=10,
        description="Default storage quota per model in GB"
    )
    
    # Vector dimension limit
    max_vector_dimension: int = Field(
        default=2000,
        description="Maximum vector dimension for models. Cloud default is 2000 (pgvector HNSW index limit). "
                    "Local installs can increase this but will lose index-based similarity search."
    )

    @property
    def resolved_max_vector_dimension(self) -> int:
        """Return effective dimension limit based on storage backend.

        SQLite stores vectors as JSON text — no size constraint applies.
        pgvector HNSW index supports up to 2000 dims; brute-force up to 16384.
        Use max_vector_dimension (env: MAX_VECTOR_DIMENSION) to override for
        pgvector deployments that don't need the HNSW index.
        """
        if self.resolved_storage_backend == "sqlite":
            return 65536  # No practical limit for SQLite JSON storage
        return self.max_vector_dimension
    
    # Model storage
    model_storage_path: str = Field(
        default="/data/models",
        description="Persistent directory for model data"
    )
    
    # Edge generation
    edge_generation_strategy: Literal["eager", "lazy", "on-demand"] = Field(
        default="lazy",
        description="Edge generation strategy"
    )
    edge_ttl_hours: int = Field(
        default=24,
        description="Edge cache TTL in hours"
    )
    
    # Beam search defaults
    default_beam_width: int = Field(default=5, description="Default beam width")
    default_max_tree_depth: int = Field(default=3, description="Default max tree depth")
    
    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    
    # Telemetry
    telemetry_enabled: bool = Field(default=True, description="Enable telemetry")
    telemetry_endpoint: Optional[str] = Field(
        default=None,
        description="Telemetry endpoint URL"
    )
    
    # CORS origins for production
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:4321",
            "http://localhost:5173",
        ],
        description="Allowed CORS origins (used when cors_allow_all is False)"
    )
    cors_origins_production: List[str] = Field(
        default=[],
        description="Allowed CORS origins for production deployments"
    )
    
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra env vars not defined in Settings


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


def validate_settings() -> None:
    """Validate settings on startup - raises if invalid."""
    settings = get_settings()

    # No validation needed — SQLite auto-defaults when DATABASE_URL is unset.
    # Add future startup checks here as needed.
