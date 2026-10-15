import pytest
from app.core.exceptions import ValidationException
from app.services.query_service import QueryProcessingService


def test_query_processor_success():
    processor = QueryProcessingService(min_length=3, max_length=500)
    result = processor.process("  What   are the   rules for code   reviews?  \n")
    assert result.cleaned_query == "What are the rules for code reviews?"
    assert "code" in result.keywords
    assert "reviews" in result.keywords
    assert "the" not in result.keywords


def test_query_processor_empty():
    processor = QueryProcessingService()
    with pytest.raises(ValidationException, match="Question must be a non-empty string"):
        processor.process("")

    with pytest.raises(ValidationException, match="Question must be a non-empty string"):
        processor.process(None)


def test_query_processor_whitespace_only():
    processor = QueryProcessingService(min_length=2)
    with pytest.raises(ValidationException, match="Question is too short"):
        processor.process("     ")


def test_query_processor_too_long():
    processor = QueryProcessingService(min_length=2, max_length=20)
    with pytest.raises(ValidationException, match="exceeds maximum permitted length"):
        processor.process("This question is much longer than twenty characters.")
