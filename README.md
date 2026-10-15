# AI Knowledge System

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![Docker](https://img.shields.io/badge/Docker-Compose%20v2-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An internal enterprise knowledge assistant designed to answer questions reliably using company documents while strictly prioritizing truthfulness, citation accuracy, and hallucination prevention over plausibility.

---

## Project Status: Phase 7 (Contradictory Knowledge and Document Version Handling)

> [!NOTE]
> This repository is currently on **Phase 8: Automated RAG Evaluation using Ragas**.
>
> The system incorporates an objective, automated evaluation harness (`python -m evaluation.run`) assessing RAG quality across 4 core metrics: **Faithfulness**, **Answer Relevancy**, **Context Precision@K**, and **Context Recall**.
>
> Rather than relying on manual checks, the benchmark systematically tests 6 query archetypes (straightforward, multi-document, unsupported, ambiguous, versioned, and contradictory knowledge) and runs comparative experiments between retrieval configurations (e.g., `top_k = 3` vs. `top_k = 5`).

---

## Key Features

- **Contradiction Analysis Engine**: Pairwise semantic analysis across retrieved chunks detecting numeric discrepancies, polarity mismatches, and schedule divergences.
- **Deterministic Conflict Resolution**: Multi-tier resolution preferring `ACTIVE` over `SUPERSEDED`/`EXPIRED`, effective dates over lapsed dates, and newer versions over older versions.
- **Anti-Vector Trap Guarding**: Strictly forbids picking a contradictory document simply due to high vector cosine similarity or keyword stuffing.
- **Explicit Conflict Transparency**: Surfaces conflicting source citations and structured `ContradictionReport` diagnostic details whenever unresolvable policy conflicts exist.
- **Document Lifecycle Metadata**: Full tracking of `version`, `status` (`ACTIVE`, `SUPERSEDED`, `EXPIRED`), `effective_date`, and `expiration_date`.
- **Multi-Stage Reliability Pipeline**: Full verification lifecycle featuring `EvidenceEvaluator`, `AnswerVerifier`, and `ReliabilityDecisionEngine`.
- **Pre-Generation Evidence Gating**: Inspects retrieval density, similarity thresholds, and query keyword coverage before invoking LLM generation, short-circuiting unsupported queries to `INSUFFICIENT_EVIDENCE`.
- **Post-Generation Claim Verification**: Decomposes answers into discrete factual claims and checks quantitative, numeric, and conceptual alignment against source documents.
- **Audited Faithfulness Score**: True mathematical fraction of supported claims ($0.0$ to $1.0$) rather than heuristic or random confidence numbers.
- **Corrective Regeneration Loop**: When unverified assertions are detected, the system attempts corrective regeneration with explicit negative boundaries before falling back to refusal.
- **Explicit Hallucination Refusal**: Strictly prefers `"I don't have enough reliable information to answer this."` with refusal reason `UNVERIFIED_CLAIMS` when evidence is inadequate.
- **Traceable Source Citations**: Every answer references exact source chunks with `document` name, `version`, `chunk_id`, `score`, and concise relevant `excerpt`.
- **Explainable Confidence Scoring**: Fully deterministic, mathematically documented confidence engine combining top similarity ($w_1=0.50$), mean cluster strength ($w_2=0.30$), and lexical keyword coverage ($w_3=0.20$).
- **Provider-Agnostic LLM Interface**: Local Ollama (`llama3:latest`), OpenAI-compatible remote gateways, and Mock testing providers.
- **Dense Vector Search**: PostgreSQL + `pgvector` with HNSW indexing and FastEmbed ONNX local models.
- **Multi-Format Ingestion**: Native parsing for PDF, DOCX, TXT, and Markdown with SHA-256 deduplication.
- **Comprehensive Pytest Suite**: 123 unit, integration, adversarial, contradiction, and evaluation tests passing both on host and in Docker container.

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
├── evaluation/                 # Automated RAG evaluation harness
│   ├── datasets/               # Benchmark dataset schemas & 11 test scenarios
│   ├── metrics/                # Faithfulness, Relevancy, Context Precision/Recall
│   ├── runners/                # EvaluationRunner & ExperimentComparator
│   ├── reports/                # Markdown report generator & JSON persistence
│   └── run.py                  # CLI evaluation runner (python -m evaluation.run)
├── tests/                      # Automated test suite
│   ├── conftest.py             # Test fixtures & async client
│   ├── unit/                   # Unit tests (config, abstractions, exceptions, repos, eval)
│   └── integration/            # Integration tests (health, search, ask, contradictions, eval)
├── scripts/                    # Automation scripts
│   ├── init_db.py              # Database connection retry & schema sync
│   └── run_dev.sh              # Local development runner
├── docker/                     # Docker infrastructure
│   ├── Dockerfile              # Production Python 3.11 container
│   └── entrypoint.sh           # Container entrypoint with DB wait
├── alembic/                    # Database migrations
├── docs/                       # Architecture & development documentation
│   ├── ARCHITECTURE.md         # Design decisions and subsystem rationale
│   ├── DEVELOPMENT.md          # Developer onboarding & guides
│   ├── CONTRADICTION_AND_VERSIONING.md # Phase 7 contradiction handling
│   └── RAG_EVALUATION.md       # Phase 8 automated evaluation guide
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

### `POST /api/v1/search`
Performs cosine similarity search using dense vector embeddings in PostgreSQL (pgvector).

**Request:**
```json
{
  "query": "data retention rules and policy",
  "top_k": 3,
  "similarity_threshold": 0.5
}
```

### `POST /api/v1/ask`
Executes an end-to-end Retrieval-Augmented Generation question-answering pipeline.

**Request:**
```json
{
  "question": "How long must customer personal identification data be retained?"
}
```

**Sample 200 OK Response (Supported Question):**
```json
{
  "question": "How long must customer personal identification data be retained?",
  "answer": "Based solely on the context provided, customer personal identification data must be retained for at least 7 years pursuant to regulatory requirements [Document 1].",
  "confidence": "high",
  "confidence_score": 0.8367,
  "reason": null,
  "sources": [
    {
      "document": "sample.docx",
      "version": "1.0",
      "chunk_id": "9377674e-b7fe-49f6-8fa5-127f6c136ada",
      "score": 0.9256,
      "excerpt": "Company Data Retention Policy Customer personal identification data must be retained for at least 7 years pursuant to regulatory requirements."
    }
  ],
  "model_name": "llama3:latest"
}
```

**Sample 200 OK Response (Insufficient Evidence / Unsupported Question):**
```json
{
  "question": "What is the company policy on bringing dogs into the office?",
  "answer": "I couldn't find sufficient information in the available knowledge base to answer this question reliably.",
  "confidence": "none",
  "confidence_score": 0.0,
  "reason": "INSUFFICIENT_EVIDENCE",
  "sources": [],
  "model_name": "llama3:latest"
}
```

### `DELETE /api/v1/documents/{document_id}`
Deletes document and cascades deletion to all associated chunks and embeddings.

### Error Responses
All errors adhere to a structured, consistent format with request tracing:
```json
{
  "success": false,
  "error": {
    "code": "LLM_GENERATION_ERROR",
    "message": "Cannot connect to Ollama service at http://localhost:11434.",
    "details": {
      "error": "Connection refused"
    }
  },
  "request_id": "f74a3012-490f-4e04-968e-8e425c6e2ddc",
  "timestamp": "2026-10-15T16:28:54.333628Z"
}
```

---

## Running Automated Tests

```bash
# Run full test suite (76 tests)
.venv/bin/pytest -v

# Run with test coverage
.venv/bin/pytest --cov=app tests/

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
| `EMBEDDING_PROVIDER` | `local` | Embedding provider: `local` (FastEmbed), `api`/`openai`, `mock` |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Dense embedding model |
| `LLM_PROVIDER` | `ollama` | Generation provider: `ollama` (local), `openai`/`api`, `mock` |
| `LLM_MODEL` | `llama3:latest` | Model identifier for Ollama or OpenAI |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama service HTTP endpoint |
| `RAG_DEFAULT_TOP_K` | `4` | Default number of retrieved chunks |
| `RAG_DEFAULT_SIMILARITY_THRESHOLD` | `0.45` | Minimum cosine similarity threshold |
| `RAG_HIGH_CONFIDENCE_THRESHOLD` | `0.75` | Minimum score for high confidence |
| `RAG_MEDIUM_CONFIDENCE_THRESHOLD` | `0.55` | Minimum score for medium confidence |
| `RAG_MAX_CONTEXT_CHARS` | `8000` | Context token/char limit |
| `RAG_EXCERPT_MAX_CHARS` | `250` | Character limit for source citation excerpts |
| `POSTGRES_SERVER` | `localhost` (or `db`) | PostgreSQL hostname |
| `POSTGRES_PORT` | `5433` (or `5432`) | PostgreSQL port |
| `POSTGRES_USER` | `postgres` | Database username |
| `POSTGRES_PASSWORD` | `postgres` | Database password |
| `POSTGRES_DB` | `ai_knowledge_db` | Database name |
| `LOG_LEVEL` | `INFO` | Logger verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_FORMAT` | `text` | Log format: `text` (human-readable) or `json` (structured) |
| `ALLOWED_ORIGINS` | `["*"]` | Allowed CORS origins (JSON array or comma-separated) |

---

## Roadmap

- [x] **Phase 1: Project Foundation** (Architecture, Docker, FastAPI, PostgreSQL, Health, Abstractions, Pytest)
- [x] **Phase 2: Document Ingestion Pipeline** (PDF/DOCX/TXT/MD, Duplicate Detection, Chunking, DB Persistence)
- [x] **Phase 3: Embeddings & Vector Database** (Local FastEmbed, pgvector HNSW Indexing, Semantic Search)
- [x] **Phase 4: Retrieval-Augmented Generation** (Query Processing, Ollama/OpenAI Abstractions, Grounded QA, Source Citations, Eval Dataset)
- [x] **Phase 5: Evidence-Based Answering & Source Citations** (Traceable Citations, Explainable Confidence Scoring, Machine-Readable Refusals, Excerpt Extraction)
- [ ] **Phase 6: Answer Faithfulness Verification & Hallucination Prevention**
- [ ] **Phase 7: Ragas Evaluation & Telemetry Dashboard**
- [ ] **Phase 8: Frontend Knowledge Assistant UI**
