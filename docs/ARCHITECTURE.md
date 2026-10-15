# AI Knowledge System - Architecture Design Document

## 1. Executive Summary

The **AI Knowledge System** is an enterprise-grade knowledge assistant designed to provide accurate, reliable answers from internal documents. Unlike conventional RAG chatbots, the system prioritizes reliability, source traceability, and hallucination prevention over plausible-sounding generation.

Phase 1 establishes the foundational architecture: a modular, loosely-coupled backend built with FastAPI, SQLAlchemy 2.0 (async), PostgreSQL with pgvector readiness, and Docker.

---

## 2. Architectural Principles

1. **Separation of Concerns & Layered Design**:
   - **Presentation Layer (`app/api/`)**: Versioned API routers (`/api/v1`), endpoint controllers, request/response serialization, and dependency injection.
   - **Domain / Service Layer (`app/services/`)**: Business workflows and domain rules (e.g., health status aggregation, document processing orchestration).
   - **Data Access Layer (`app/repositories/`)**: Abstracted CRUD repository pattern isolating database queries from domain logic.
   - **Data Layer (`app/models/`, `app/core/database.py`)**: SQLAlchemy DeclarativeBase ORM models and asynchronous connection pooling.

2. **Interface Abstraction & Loose Coupling**:
   - Every AI and pipeline component is represented by an Abstract Base Class (ABC) contract.
   - Future implementations can be swapped without touching the core API or domain services:
     - **LLM Provider (`app/generation/base.py`)**: `BaseLLMProvider` decouples local models (Ollama, vLLM) from hosted APIs (OpenAI, Anthropic, Gemini).
     - **Vector Store (`app/retrieval/base.py`)**: `BaseVectorStore` isolates pgvector, Chroma, Qdrant, or cloud vector databases.
     - **Ingestion (`app/ingestion/base.py`)**: `BaseDocumentParser`, `BaseChunker`, and `BaseEmbedder` allow pluggable document formats and chunking strategies.
     - **Evaluation (`app/evaluation/base.py`)**: `BaseEvaluator` interfaces directly with Ragas or LLM-as-judge metrics.

3. **Production Observability by Default**:
   - Correlation IDs: Every request receives or generates a unique `X-Request-ID` attached to logs, request state, and HTTP headers.
   - Execution Profiling: `X-Process-Time` is calculated and returned on every response.
   - Dual-Mode Logging: Human-readable colored output in development and structured JSON output in production.

4. **12-Factor Configuration & Secret Hygiene**:
   - All settings are defined in `app/core/config.py` using Pydantic Settings.
   - No credentials or API keys are hardcoded in the codebase.
   - Safe fallbacks and non-conflicting host port defaults ensure smooth local execution.

---

## 3. High-Level System Architecture

```
+-------------------------------------------------------------------------+
|                              Client Layer                               |
|                  (Browser / API Consumers / Future React UI)            |
+-------------------------------------------------------------------------+
                                     |
                                     | HTTP / JSON (X-Request-ID)
                                     v
+-------------------------------------------------------------------------+
|                           FastAPI Application                           |
|                                                                         |
|  [Middleware]                                                           |
|    - CORS Policy                                                        |
|    - RequestContextMiddleware (Tracing, Timing, Correlation ID)         |
|    - Structured Exception Handlers (AppException -> Standard JSON)      |
|                                                                         |
|  [Routers: /api/v1]                                                     |
|    - GET /api/v1/health (Database ping, System metrics, Uptime)         |
|                                                                         |
|  [Application Services]                                                 |
|    - HealthService                                                      |
+-------------------------------------------------------------------------+
       |                         |                          |
       v                         v                          v
+---------------+      +-------------------+      +-------------------+
| Repositories  |      |   AI Subsystems   |      |   Observability   |
| (BaseRepo)    |      |   (Abstractions)  |      | (JSON Log, Time)  |
+---------------+      +-------------------+      +-------------------+
       |                   - Ingestion                - Logs / stdout
       |                   - Retrieval                - X-Request-ID
       |                   - Generation               - Metrics hooks
       |                   - Evaluation
       v
+-------------------------------------------------------------------------+
|                           Infrastructure                                |
|   - PostgreSQL 16 with pgvector (ai_knowledge_db)                       |
|   - Docker Compose Multi-Container Orchestration                        |
+-------------------------------------------------------------------------+
```

---

## 4. Module Layout

```
AiKnowledgeSystem/
├── app/
│   ├── api/                    # API presentation layer
│   │   ├── dependencies.py     # FastAPI dependency injection
│   │   └── v1/
│   │       ├── router.py       # API v1 central router
│   │       └── endpoints/
│   │           └── health.py   # Health & readiness endpoint
│   ├── core/                   # Application core infrastructure
│   │   ├── config.py           # Pydantic Settings & environment validation
│   │   ├── database.py         # Async SQLAlchemy engine & session maker
│   │   ├── exceptions.py       # Custom exceptions & global handlers
│   │   └── logging.py          # Structured JSON & console logger
│   ├── models/                 # ORM domain models
│   │   └── base.py             # Base model mixins (UUID, Timestamps)
│   ├── schemas/                # Pydantic DTOs & API schemas
│   │   ├── common.py           # Standard response & error envelopes
│   │   └── health.py           # Health response schemas
│   ├── repositories/           # Generic data access pattern
│   │   └── base.py             # BaseRepository[ModelType]
│   ├── services/               # Business logic services
│   │   └── health_service.py   # Health calculation & verification
│   ├── ingestion/              # Ingestion contracts
│   │   └── base.py             # BaseDocumentParser, BaseChunker, BaseEmbedder
│   ├── retrieval/              # Retrieval contracts
│   │   └── base.py             # BaseVectorStore, BaseRetriever
│   ├── generation/             # Generation contracts
│   │   └── base.py             # BaseLLMProvider, LLMResponse
│   ├── evaluation/             # Evaluation contracts
│   │   └── base.py             # BaseEvaluator, RAGEvaluationReport
│   ├── observability/          # Telemetry & middleware
│   │   └── telemetry.py        # RequestContextMiddleware
│   └── main.py                 # FastAPI application factory & lifespan
├── tests/                      # Automated test suite
│   ├── conftest.py             # Pytest fixtures & async client
│   ├── unit/                   # Isolated unit tests
│   └── integration/            # API integration tests
├── scripts/                    # Automation & helper scripts
│   ├── init_db.py              # DB connection wait & schema init
│   └── run_dev.sh              # Local development server script
├── docker/                     # Container configuration
│   ├── Dockerfile              # Production Python 3.11 container
│   └── entrypoint.sh           # DB wait & startup entrypoint
├── docker-compose.yml          # Container orchestration (Postgres + Backend)
├── alembic/                    # Database migration environment
├── .env.example                # Sample environment template
└── pyproject.toml              # Tooling configuration
```

---

## 5. Architectural Decisions & Rationale

| Decision | Choice | Rationale |
|---|---|---|
| **Web Framework** | FastAPI | High async performance, automatic OpenAPI schema generation, native Pydantic v2 validation. |
| **Database ORM** | SQLAlchemy 2.0 (Async) | Industry standard, robust connection pooling, full async/await support with `asyncpg`. |
| **Vector Storage** | `pgvector/pgvector:pg16` | Keeps relational and vector data co-located, reducing infrastructure complexity for Phase 1 and 2. |
| **Interfaces (ABC)** | Python `abc.ABC` | Strict contract enforcement during development so RAG components can be swapped without breakage. |
| **Testing Strategy** | Pytest + Async SQLite | Fast, isolated test execution without mandatory external Postgres during CI / local test cycles. |
| **Port Mapping** | App: `8001`, DB: `5433` | Configurable with defaults that prevent port collisions on busy developer workstations. |
