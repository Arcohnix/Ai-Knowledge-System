# Development Guide

This guide details setting up, developing, testing, and managing the **AI Knowledge System**.

---

## Prerequisites

- **Python**: `>= 3.11` (Python 3.11 - 3.13 supported)
- **Docker & Docker Compose**: v2+
- **Virtualenv** / Package manager: `venv` or `pipenv`

---

## Local Setup

### 1. Clone & Environment Setup

```bash
# Copy example environment variables
cp .env.example .env

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install all dependencies (including dev/test)
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 2. Running with Docker Compose (Recommended)

Starts both the PostgreSQL (with pgvector) database and the FastAPI backend service:

```bash
docker compose up --build
```

- **Backend API**: `http://localhost:8001`
- **Swagger Docs**: `http://localhost:8001/docs`
- **Health Check**: `http://localhost:8001/api/v1/health`
- **Postgres DB**: `localhost:5433` (`ai_knowledge_db`)

To shut down:
```bash
docker compose down
```

To shut down and wipe persistent volumes:
```bash
docker compose down -v
```

---

## Running Locally Without Docker

If you prefer to run the FastAPI app directly on your host:

```bash
# Ensure your virtual environment is active
source .venv/bin/activate

# Start server using the runner script
./scripts/run_dev.sh

# Or start directly using uvicorn:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Running Tests

All tests run via pytest with async support:

```bash
# Run all tests
pytest -v

# Run tests with code coverage report
pytest --cov=app tests/

# Run specific test file
pytest tests/integration/test_health.py -v
```

---

## Database Migrations (Alembic)

To generate a new migration revision:
```bash
alembic revision --autogenerate -m "create_initial_tables"
```

To apply migrations:
```bash
alembic upgrade head
```

To roll back one revision:
```bash
alembic downgrade -1
```

---

## Code Structure Conventions

- **Adding endpoints**: Place routers under `app/api/v1/endpoints/` and register them in `app/api/v1/router.py`.
- **Adding database models**: Inherit from `Base` in `app/models/base.py` and register the model import in `app/models/__init__.py`.
- **Adding services**: Put business logic in `app/services/` and inject repositories or external clients.
- **Handling errors**: Raise custom exceptions derived from `AppException` in `app/core/exceptions.py`.
