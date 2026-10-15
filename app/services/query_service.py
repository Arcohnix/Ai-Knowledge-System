import re
import unicodedata
from typing import List, Set
from pydantic import BaseModel, Field
from app.core.exceptions import ValidationException
from app.core.logging import get_logger

logger = get_logger(__name__)

STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "from",
    "by", "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "of", "in", "on", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "can",
    "could", "should", "would", "may", "might", "must", "what", "which", "who", "whom",
    "this", "that", "these", "those", "am", "how", "why"
}


class ProcessedQuery(BaseModel):
    """Encapsulates the validated and normalized query string with extracted keywords."""

    raw_query: str = Field(..., description="Original raw question string")
    cleaned_query: str = Field(..., description="Sanitized and normalized query text")
    keywords: List[str] = Field(
        default_factory=list, description="Extracted semantic keywords"
    )


class QueryProcessingService:
    """
    Dedicated service for query validation, sanitization, and normalization.
    Ensures input safety and removes extraneous formatting noise.
    """

    def __init__(self, min_length: int = 2, max_length: int = 2000):
        self.min_length = min_length
        self.max_length = max_length

    def process(self, query: str) -> ProcessedQuery:
        """
        Validates, sanitizes, and normalizes an input question.
        """
        if not query or not isinstance(query, str):
            raise ValidationException("Question must be a non-empty string.")

        # Normalize unicode (NFKC)
        normalized = unicodedata.normalize("NFKC", query)

        # Strip non-printable and control characters
        cleaned = "".join(
            ch for ch in normalized if unicodedata.category(ch)[0] != "C" or ch in ("\n", " ")
        )

        # Collapse whitespace and trim
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if len(cleaned) < self.min_length:
            raise ValidationException(
                f"Question is too short after normalization (minimum {self.min_length} characters required)."
            )

        if len(cleaned) > self.max_length:
            raise ValidationException(
                f"Question exceeds maximum permitted length of {self.max_length} characters."
            )

        # Extract meaningful keywords for telemetry and debugging
        tokens = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", cleaned.lower())
        keywords = [token for token in tokens if token not in STOP_WORDS]

        logger.debug(
            f"Query processed: raw_len={len(query)}, clean_len={len(cleaned)}, keywords={keywords[:6]}"
        )

        return ProcessedQuery(
            raw_query=query,
            cleaned_query=cleaned,
            keywords=keywords,
        )
