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
> This repository is currently on **Phase 2: Document Ingestion Pipeline**.
>
> Core foundation and document ingestion are operational: files (PDF, DOCX, TXT, Markdown) are validated, parsed, cleaned, intelligently chunked, and stored with rich metadata and duplicate detection. **No embeddings or vector search implemented yet.**

---

## Key Features

- **Multi-Format Ingestion**: Native parsing support for **PDF** (`pypdf`), **DOCX** (`python-docx`), **Plain Text** (`.txt`), and **Markdown** (`.md`).
- **Duplicate Document Detection**: SHA-256 content hashing blocks duplicate uploads with a structured 409 Conflict response.
- **Sensible Structural Chunking**: Recursive hierarchical chunking respecting markdown headings, paragraph breaks (`\n\n`), sentence boundaries, and words with configurable `chunk_size` and `chunk_overlap`.
- **Text Normalization & Cleaning**: Strips null bytes, non-printable control codes, normalizes Unicode (NFKC), and standardizes line breaks.
- **Relational Persistence**: PostgreSQL tables for `documents` and `document_chunks` with cascade deletion, indexed foreign keys, and JSON metadata.
- **Clean Layered Architecture**: Strictly separated presentation (`api`), domain logic (`services`), data access (`repositories`), and core infrastructure (`core`).
- **Pluggable AI Abstractions**: Abstract Base Classes ready for swappable LLM providers (Ollama / OpenAI / Claude / Gemini), Vector Stores (pgvector / Chroma / Qdrant), Document Parsers, and Evaluators.
- **Production Observability**: Correlation IDs (`X-Request-ID`) and request timing (`X-Process-Time`) on every request.
- **Comprehensive Pytest Suite**: 41 unit and integration tests passing both on host and inside Docker container.

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

### `POST /api/v1/documents`
Ingests an uploaded document (`multipart/form-data`). Supported formats: `.pdf`, `.docx`, `.txt`, `.md`, `.markdown`.

**Parameters:**
- `file`: UploadFile (required)
- `chunk_size`: int (optional, default: 500)
- `chunk_overlap`: int (optional, default: 50)
- `source`: str (optional, default: "upload")

**Sample 201 Created Response:**
```json
{
  "document_id": "18c6e5d8-5fef-4219-b323-8a58bb336b0e",
  "filename": "security.md",
  "document_type": "markdown",
  "title": "Enterprise Security Architecture",
  "source": "upload",
  "version": "1.0",
  "checksum": "a7cbd8f27558917f31caf45230914024165930d256f13295de3440144dcc50de",
  "metadata": {
    "source_type": "markdown",
    "filename": "security.md",
    "character_count": 218,
    "line_count": 7,
    "file_size_bytes": 218,
    "chunk_count": 1
  },
  "created_at": "2026-10-15T16:28:48.821839Z",
  "updated_at": "2026-10-15T16:28:48.821842Z",
  "chunk_count": 1
}
```

### `GET /api/v1/documents`
Returns paginated list of ingested documents with optional `document_type` filtering.

**Query Parameters:**
- `skip`: int (default: 0)
- `limit`: int (default: 20)
- `document_type`: str (optional, e.g. `pdf`, `docx`, `txt`, `markdown`)

### `GET /api/v1/documents/{document_id}`
Returns document metadata alongside all extracted text chunks.

### `DELETE /api/v1/documents/{document_id}`
Deletes document and cascades deletion to all associated chunks.

### Error Responses
All errors adhere to a structured, consistent format with request tracing:
```json
{
  "success": false,
  "error": {
    "code": "CONFLICT",
    "message": "Duplicate document detected: 'security.md' has already been ingested.",
    "details": {
      "existing_document_id": "18c6e5d8-5fef-4219-b323-8a58bb336b0e",
      "existing_filename": "security.md",
      "checksum": "a7cbd8f27558917f31caf45230914024165930d256f13295de3440144dcc50de"
    }
  },
  "request_id": "f74a3012-490f-4e04-968e-8e425c6e2ddc",
  "timestamp": "2026-10-15T16:28:54.333628Z"
}
```

---

## Running Automated Tests

```bash
# Run full test suite (41 tests)
pytest -v

# Run with test coverage
pytest --cov=app tests/

# Run tests inside Docker container
docker compose exec backend pytest -v
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `development` | Deployment environment (`development`, `staging`, `production`) |
| `DEBUG` | `true` | Enables FastAPI debug mode and SQL query logging |
| `APP_HOST_PORT` | `8001` | Host port mapped to FastAPI container in Docker Compose |
| `DB_HOST_PORT` | `5433` | Host port mapped to PostgreSQL container in Docker Compose |
| `DEFAULT_CHUNK_SIZE` | `500` | Target character size per text chunk |
| `DEFAULT_CHUNK_OVERLAP`| `50` | Character overlap between consecutive chunks |
| `MAX_FILE_SIZE_BYTES` | `26214400` | Maximum allowed file upload size (25 MB) |
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
- [x] **Phase 2: Document Ingestion Pipeline** (PDF/DOCX/TXT/MD, Duplicate Detection, Chunking, DB Persistence)
- [ ] **Phase 3: Chunking & Dense Embeddings**
- [ ] **Phase 4: Vector Store & Retrieval Pipeline**
- [ ] **Phase 5: Generation, Hallucination Prevention & "I don't know" Logic**
- [ ] **Phase 6: Faithfulness Verification & Quality Checks**
- [ ] **Phase 7: Ragas Evaluation & Telemetry Dashboard**
- [ ] **Phase 8: Frontend Knowledge Assistant UI**
