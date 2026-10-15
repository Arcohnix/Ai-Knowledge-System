# AI Knowledge System

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![Docker](https://img.shields.io/badge/Docker-Compose%20v2-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An internal enterprise knowledge assistant designed to answer questions reliably using company documents while strictly prioritizing truthfulness, citation accuracy, and hallucination prevention over plausibility.

---

## Project Status: Phase 1 (Foundation)

> [!NOTE]
> This repository is currently in **Phase 1: Project Foundation**.
>
> Core infrastructure, clean layered architecture, database connectivity, Docker orchestration, abstract component contracts (LLM, Vector Store, Ingestion, Evaluation), and health endpoints are operational. **No RAG functionality is implemented yet.**

---

## Key Features (Phase 1)

- **Clean Layered Architecture**: Strictly separated presentation (`api`), domain logic (`services`), data access (`repositories`), and core infrastructure (`core`).
- **Pluggable AI Abstractions**: Abstract Base Classes ready for swappable LLM providers (Ollama / OpenAI / Claude / Gemini), Vector Stores (pgvector / Chroma / Qdrant), Document Parsers, and Evaluators.
- **Async SQLAlchemy 2.0 & PostgreSQL**: Modern async connection pooling (`asyncpg`), metadata models, and Alembic migrations.
- **Production Observability**:
  - Structured JSON logging for production, readable colored logs for development.
  - Correlation IDs (`X-Request-ID`) generated or passed through on every request.
  - Latency tracking (`X-Process-Time`) measured and returned in HTTP headers.
- **Structured Error Handling**: Centralized exception handlers converting custom domain exceptions into standardized, consistent JSON envelopes.
- **Health & Readiness Check**: `GET /api/v1/health` dynamically pings PostgreSQL, checks roundtrip query latency, and reports system runtime status.
- **12-Factor Configuration**: Fully typed settings via `pydantic-settings` with zero hardcoded credentials.
- **Dockerized Multi-Container Setup**: One-command startup orchestrating the FastAPI backend and PostgreSQL with `pgvector`.
- **Comprehensive Pytest Suite**: Async unit and integration tests with 100% pass rate.

---

## Architecture Overview

```
AiKnowledgeSystem/
├── app/
│   ├── api/                    # API routes, v1 versioning, dependency injection
│   │   ├── dependencies.py     # Database session injection
│   │   └── v1/
│   │       ├── router.py       # API v1 central router
│   │       └── endpoints/
│   │           └── health.py   # Health endpoint (GET /api/v1/health)
│   ├── core/                   # Infrastructure configuration
│   │   ├── config.py           # Pydantic Settings & env parsing
│   │   ├── database.py         # Async engine, sessionmaker, ping utility
│   │   ├── exceptions.py       # Custom AppException hierarchy & global handlers
│   │   └── logging.py          # Structured JSON / Text loggers
│   ├── models/                 # SQLAlchemy 2.0 ORM models
│   │   └── base.py             # DeclarativeBase, UUID & Timestamp mixins
│   ├── schemas/                # Pydantic request/response schemas
│   │   ├── common.py           # APIResponse, ErrorResponse
│   │   └── health.py           # HealthCheckResponse & component schemas
│   ├── repositories/           # Abstract CRUD repository pattern
│   │   └── base.py             # Generic BaseRepository[ModelType]
│   ├── services/               # Business domain services
│   │   └── health_service.py   # System health verification service
│   ├── ingestion/              # Ingestion abstraction contracts
│   │   └── base.py             # BaseDocumentParser, BaseChunker, BaseEmbedder
│   ├── retrieval/              # Retrieval abstraction contracts
│   │   └── base.py             # BaseVectorStore, BaseRetriever, RetrievedChunk
│   ├── generation/             # Generation abstraction contracts
│   │   └── base.py             # BaseLLMProvider, LLMResponse
│   ├── evaluation/             # Evaluation abstraction contracts
│   │   └── base.py             # BaseEvaluator, RAGEvaluationReport
│   ├── observability/          # Tracing & telemetry
│   │   └── telemetry.py        # RequestContextMiddleware (X-Request-ID, Timing)
│   └── main.py                 # FastAPI application factory & lifespan
├── tests/                      # Automated test suite
│   ├── conftest.py             # Test fixtures & async client
│   ├── unit/                   # Unit tests (config, abstractions, exceptions, repos)
│   └── integration/            # Integration tests (health endpoint, middleware)
├── scripts/                    # Automation scripts
│   ├── init_db.py              # Database connection retry & schema sync
│   └── run_dev.sh              # Local development runner
├── docker/                     # Docker infrastructure
│   ├── Dockerfile              # Production Python 3.11 container
│   └── entrypoint.sh           # Container entrypoint with DB wait
├── alembic/                    # Database migrations
├── docs/                       # Architecture & development documentation
│   ├── ARCHITECTURE.md         # Design decisions and subsystem rationale
│   └── DEVELOPMENT.md          # Developer onboarding & guides
├── docker-compose.yml          # Container orchestration (PostgreSQL + FastAPI)
├── .env.example                # Environment variable template
└── pyproject.toml              # Project metadata & pytest configuration
```

---

## Quickstart

### Option 1: Run with Docker Compose (Recommended)

1. **Clone the repository and prepare `.env`**:
   ```bash
   cp .env.example .env
   ```

2. **Start all services**:
   ```bash
   docker compose up --build
   ```

3. **Verify the services**:
   - Backend API: [http://localhost:8001](http://localhost:8001)
   - Interactive Swagger Docs: [http://localhost:8001/docs](http://localhost:8001/docs)
   - Health Check Endpoint: [http://localhost:8001/api/v1/health](http://localhost:8001/api/v1/health)
   - PostgreSQL: `localhost:5433` (database: `ai_knowledge_db`)

4. **Stop the services**:
   ```bash
   docker compose down
   ```

---

### Option 2: Run Locally (Host Machine)

1. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements-dev.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   ```

4. **Start the application**:
   ```bash
   ./scripts/run_dev.sh
   # Or directly:
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## API Reference (Phase 1)

### `GET /api/v1/health`
Returns runtime status and checks live PostgreSQL database connectivity.

**Sample 200 OK Response:**
```json
{
  "status": "healthy",
  "app_name": "AI Knowledge System",
  "version": "0.1.0",
  "environment": "development",
  "uptime_seconds": 12.45,
  "timestamp": "2026-10-15T16:05:00.123456Z",
  "components": {
    "database": {
      "status": "healthy",
      "latency_ms": 1.82,
      "error": null
    },
    "system": {
      "python_version": "3.11.10",
      "platform": "Linux-x86_64"
    }
  }
}
```

### `GET /`
Returns service root information and discovery links.

**Sample 200 OK Response:**
```json
{
  "name": "AI Knowledge System",
  "version": "0.1.0",
  "environment": "development",
  "docs": "/docs",
  "health": "/api/v1/health"
}
```

### Error Responses
All errors adhere to a structured, consistent format with request tracing:
```json
{
  "success": false,
  "error": {
    "code": "NOT_FOUND",
    "message": "Resource not found",
    "details": null
  },
  "request_id": "713f01c8-a968-466d-97a7-58fb177241cf",
  "timestamp": "2026-10-15T16:05:01.000000Z"
}
```

---

## Running Automated Tests

```bash
# Run full test suite
pytest -v

# Run with test coverage
pytest --cov=app tests/
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `development` | Deployment environment (`development`, `staging`, `production`) |
| `DEBUG` | `true` | Enables FastAPI debug mode and SQL query logging |
| `APP_HOST_PORT` | `8001` | Host port mapped to FastAPI container in Docker Compose |
| `DB_HOST_PORT` | `5433` | Host port mapped to PostgreSQL container in Docker Compose |
| `POSTGRES_SERVER` | `localhost` (or `db`) | PostgreSQL hostname |
| `POSTGRES_PORT` | `5433` (or `5432`) | PostgreSQL port |
| `POSTGRES_USER` | `postgres` | Database username |
| `POSTGRES_PASSWORD` | `postgres` | Database password |
| `POSTGRES_DB` | `ai_knowledge_db` | Database name |
| `DATABASE_URL` | Auto-constructed | Full async database connection URL |
| `LOG_LEVEL` | `INFO` | Logger verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_FORMAT` | `text` | Log format: `text` (human-readable) or `json` (structured) |
| `ALLOWED_ORIGINS` | `["*"]` | Allowed CORS origins (JSON array or comma-separated) |

---

## Roadmap

- [x] **Phase 1: Project Foundation** (Architecture, Docker, FastAPI, PostgreSQL, Health, Abstractions, Pytest)
- [ ] **Phase 2: Ingestion & Parsing Engine**
- [ ] **Phase 3: Chunking & Dense Embeddings**
- [ ] **Phase 4: Vector Store & Retrieval Pipeline**
- [ ] **Phase 5: Generation, Hallucination Prevention & "I don't know" Logic**
- [ ] **Phase 6: Faithfulness Verification & Quality Checks**
- [ ] **Phase 7: Ragas Evaluation & Telemetry Dashboard**
- [ ] **Phase 8: Frontend Knowledge Assistant UI**
