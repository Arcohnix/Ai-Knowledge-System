import json
from typing import Any, List, Optional, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Application Information
    APP_NAME: str = "AI Knowledge System"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"

    # Server Binding
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # PostgreSQL Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5433
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "ai_knowledge_db"
    DATABASE_URL: Optional[str] = None

    # Connection Pool Settings
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "text"  # 'text' or 'json'

    # Ingestion & Chunking
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50
    MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB
    SUPPORTED_DOCUMENT_TYPES: List[str] = ["pdf", "docx", "txt", "md", "markdown"]

    # Embeddings & Vector Store
    EMBEDDING_PROVIDER: str = "local"  # 'local' (fastembed), 'api' / 'openai', 'mock'
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_API_KEY: Optional[str] = None
    EMBEDDING_API_BASE_URL: str = "https://api.openai.com/v1"
    EMBEDDING_BATCH_SIZE: int = 32
    DEFAULT_SEARCH_TOP_K: int = 5
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.5

    # LLM Generation
    LLM_PROVIDER: str = "ollama"  # 'ollama', 'openai', 'mock'
    LLM_MODEL: str = "llama3.2:latest"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1024
    LLM_REQUEST_TIMEOUT: float = 60.0
    LLM_MAX_RETRIES: int = 2
    LLM_RETRY_DELAY: float = 0.5
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_BASE_URL: str = "https://api.openai.com/v1"

    # Security & Guardrails
    ENABLE_PROMPT_INJECTION_DETECTION: bool = True
    MIN_QUESTION_LENGTH: int = 3
    MAX_QUESTION_LENGTH: int = 1500

    # RAG Orchestration & Evidence Sufficiency
    RAG_FALLBACK_RESPONSE: str = (
        "I couldn't find sufficient information in the available knowledge base to answer this question reliably."
    )
    RAG_INSUFFICIENT_EVIDENCE_REASON: str = "INSUFFICIENT_EVIDENCE"
    RAG_DEFAULT_TOP_K: int = 4
    RAG_DEFAULT_SIMILARITY_THRESHOLD: float = 0.45
    RAG_HIGH_CONFIDENCE_THRESHOLD: float = 0.75
    RAG_MEDIUM_CONFIDENCE_THRESHOLD: float = 0.55
    RAG_MAX_CONTEXT_CHARS: int = 8000
    RAG_EXCERPT_MAX_CHARS: int = 250

    # Faithfulness Verification & Reliability (Phase 6)
    RAG_FAITHFULNESS_THRESHOLD: float = 0.80
    RAG_MAX_REGENERATION_ATTEMPTS: int = 1
    RAG_UNVERIFIED_REFUSAL_RESPONSE: str = (
        "I don't have enough reliable information to answer this."
    )
    RAG_UNVERIFIED_REASON: str = "UNVERIFIED_CLAIMS"
    RAG_ENABLE_REGENERATION: bool = True

    # Contradiction Detection & Document Version Handling (Phase 7)
    RAG_ENABLE_CONTRADICTION_DETECTION: bool = True
    RAG_CONFLICT_REFUSAL_RESPONSE: str = (
        "I found conflicting information in the knowledge base and cannot determine which policy is currently authoritative."
    )
    RAG_CONFLICT_REASON: str = "CONFLICTING_EVIDENCE"


    # CORS
    ALLOWED_ORIGINS: Union[List[str], str] = ["*"]

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            v_stripped = v.strip()
            if v_stripped.startswith("[") and v_stripped.endswith("]"):
                try:
                    return json.loads(v_stripped)
                except Exception:
                    pass
            return [i.strip() for i in v_stripped.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return ["*"]

    @property
    def async_database_url(self) -> str:
        """Returns the async SQLAlchemy database URL."""
        if self.DATABASE_URL and self.DATABASE_URL.strip():
            return self.DATABASE_URL.strip()
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


settings = Settings()
