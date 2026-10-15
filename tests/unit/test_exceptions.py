import pytest
from fastapi import status
from app.core.exceptions import (
    AppException,
    NotFoundError,
    ConflictError,
    ValidationException,
    DatabaseConnectionError,
    ServiceUnavailableError,
)


def test_exception_status_codes_and_defaults():
    """Verify standard application exceptions carry correct status codes and error codes."""
    not_found = NotFoundError("Item 123 missing")
    assert not_found.status_code == status.HTTP_404_NOT_FOUND
    assert not_found.error_code == "NOT_FOUND"
    assert not_found.message == "Item 123 missing"

    conflict = ConflictError("Duplicate slug")
    assert conflict.status_code == status.HTTP_409_CONFLICT
    assert conflict.error_code == "CONFLICT"

    val_err = ValidationException("Invalid email format")
    assert val_err.status_code == 422
    assert val_err.error_code == "VALIDATION_ERROR"

    db_err = DatabaseConnectionError("Timeout connecting to Postgres")
    assert db_err.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert db_err.error_code == "DATABASE_UNAVAILABLE"

    svc_err = ServiceUnavailableError("LLM API overloaded")
    assert svc_err.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert svc_err.error_code == "SERVICE_UNAVAILABLE"


def test_custom_app_exception():
    """Verify custom AppException with specific codes and extra details."""
    exc = AppException(
        message="Custom business violation",
        status_code=400,
        error_code="CUSTOM_RULE_VIOLATION",
        details={"field": "query", "reason": "too long"},
    )
    assert exc.status_code == 400
    assert exc.error_code == "CUSTOM_RULE_VIOLATION"
    assert exc.details == {"field": "query", "reason": "too long"}
