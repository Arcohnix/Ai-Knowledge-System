import pytest
from app.core.config import Settings


def test_settings_defaults():
    """Verify application settings initialize with sane defaults from components."""
    cfg = Settings(
        _env_file=None,
        DATABASE_URL=None,
        APP_NAME="Test Knowledge System",
        POSTGRES_SERVER="db.example.com",
        POSTGRES_PORT=5432,
        POSTGRES_USER="testuser",
        POSTGRES_PASSWORD="secretpassword",
        POSTGRES_DB="testdb",
    )
    assert cfg.APP_NAME == "Test Knowledge System"
    assert cfg.ENVIRONMENT in ["development", "testing", "production"]
    assert (
        cfg.async_database_url
        == "postgresql+asyncpg://testuser:secretpassword@db.example.com:5432/testdb"
    )


def test_cors_origins_parsing():
    """Verify CORS origins string and list inputs are parsed into a list of strings."""
    # List format
    cfg_list = Settings(_env_file=None, ALLOWED_ORIGINS=["http://localhost:3000"])
    assert cfg_list.ALLOWED_ORIGINS == ["http://localhost:3000"]

    # Comma-separated format
    cfg_str = Settings(
        _env_file=None, ALLOWED_ORIGINS="http://localhost:3000, http://localhost:8000"
    )
    assert cfg_str.ALLOWED_ORIGINS == ["http://localhost:3000", "http://localhost:8000"]

    # JSON array string format
    cfg_json = Settings(
        _env_file=None, ALLOWED_ORIGINS='["http://localhost:3000","http://example.com"]'
    )
    assert cfg_json.ALLOWED_ORIGINS == ["http://localhost:3000", "http://example.com"]


def test_explicit_database_url_override():
    """Verify explicit DATABASE_URL overrides the constructed Postgres URL."""
    custom_url = "postgresql+asyncpg://custom:pass@custom-host:5432/custom_db"
    cfg = Settings(_env_file=None, DATABASE_URL=custom_url)
    assert cfg.async_database_url == custom_url
