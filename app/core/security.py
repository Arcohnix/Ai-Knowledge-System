import logging
import os
import re
from typing import Optional, Tuple

logger = logging.getLogger("app.core.security")

# Common prompt injection patterns, role overrides, and system prompt leakage attempts
PROMPT_INJECTION_PATTERNS = [
    # Instruction overrides
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|rules|commands)",
    r"disregard\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|rules|guidelines)",
    r"forget\s+(?:all\s+)?(?:everything|previous|prior)\s+(?:instructions|rules)",
    r"override\s+(?:system|safety|security)\s+(?:prompt|instructions|rules)",
    # Persona / Jailbreak switches
    r"you\s+are\s+now\s+(?:dan|jailbroken|an\s+unrestricted|in\s+developer\s+mode|free\s+of\s+rules)",
    r"act\s+as\s+(?:dan|an\s+unfiltered|a\s+rogue\s+ai|an\s+evil\s+ai)",
    r"switch\s+to\s+(?:developer\s+mode|unrestricted\s+mode|god\s+mode)",
    # System prompt extraction / leakage
    r"(?:print|output|display|show|reveal|leak|repeat)\s+(?:the\s+|your\s+)?(?:(?:initial|developer|hidden|secret)\s+)?(?:system\s+prompt|prompt|instructions)",
    r"what\s+(?:is|are)\s+(?:the\s+|your\s+)?(?:(?:initial|developer|hidden|secret)\s+)?(?:system\s+prompt|prompt|instructions)",
    # Chat template delimiter injection
    r"<\|system\|>",
    r"<\|im_start\|>",
    r"\[INST\]",
    r"<<SYS>>",
    r"---\s*BEGIN\s+SYSTEM\s+PROMPT\s*---",
]

_COMPILED_INJECTION_REGEX = [
    re.compile(p, re.IGNORECASE) for p in PROMPT_INJECTION_PATTERNS
]

# Sensitive token masking patterns for logs
SENSITIVE_PATTERNS = [
    (re.compile(r"(?i)(bearer\s+)[a-zA-Z0-9_\-\.]{15,}"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(api[_\-]?key\s*[:=]\s*['\"]?)[a-zA-Z0-9_\-\.]{15,}(['\"]?)"), r"\1[REDACTED]\2"),
    (re.compile(r"(?i)(password\s*[:=]\s*['\"]?)[^'\"\s]{4,}(['\"]?)"), r"\1[REDACTED]\2"),
    (re.compile(r"(?i)(secret\s*[:=]\s*['\"]?)[a-zA-Z0-9_\-\.]{10,}(['\"]?)"), r"\1[REDACTED]\2"),
]


def detect_prompt_injection(query: str) -> Tuple[bool, Optional[str]]:
    """
    Scans an input question for obvious prompt injection, system prompt leakage,
    or jailbreak delimiter injection attempts.
    Returns:
        (is_injection: bool, matched_pattern_description: Optional[str])
    """
    if not query:
        return False, None

    normalized_query = query.strip()
    for regex in _COMPILED_INJECTION_REGEX:
        match = regex.search(normalized_query)
        if match:
            matched_text = match.group(0)
            logger.warning(
                f"Security alert: Prompt injection pattern detected: '{matched_text}'"
            )
            return True, f"Detected injection pattern: '{matched_text}'"

    return False, None


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes an uploaded filename to prevent directory traversal attacks (e.g. '../'),
    strips dangerous shell characters, and ensures a clean basename.
    """
    if not filename:
        from app.core.exceptions import ValidationException
        raise ValidationException("Filename cannot be empty")

    # Extract base filename and eliminate directory traversal (handles Unix and Windows separators)
    clean_path = filename.strip().replace("\\", "/")
    base_name = os.path.basename(clean_path)

    # Remove null bytes and control characters
    base_name = re.sub(r"[\x00-\x1f\x7f]", "", base_name)

    # Normalize dangerous path symbols and shell metacharacters
    base_name = re.sub(r"[\\/:\*\?\"<>\|]", "_", base_name)

    # Prevent hidden files
    if base_name.startswith("."):
        base_name = f"doc_{base_name.lstrip('.')}"

    if not base_name or base_name == ".":
        from app.core.exceptions import ValidationException
        raise ValidationException("Sanitized filename resulted in an empty or invalid name.")

    return base_name


def validate_file_magic(content: bytes, document_type: str) -> bool:
    """
    Verifies that the byte content matches the declared file signature / magic numbers.
    Prevents executable/binary files disguised as text or documents.
    """
    if not content:
        return False

    doc_type = document_type.lower().strip()

    if doc_type == "pdf":
        # PDF magic bytes: %PDF-
        return content.startswith(b"%PDF-")

    if doc_type == "docx":
        # DOCX is a ZIP container: PK\x03\x04
        return content.startswith(b"PK\x03\x04")

    if doc_type in ("txt", "md", "markdown"):
        # Text/markdown should not contain null bytes or excessive non-printable binary characters
        sample = content[:4096]
        if b"\x00" in sample:
            return False
        # Verify valid UTF-8 or ASCII encoding
        try:
            sample.decode("utf-8")
            return True
        except UnicodeDecodeError:
            try:
                sample.decode("latin-1")
                return True
            except UnicodeDecodeError:
                return False

    return True


def mask_sensitive_data(text: str) -> str:
    """
    Redacts API keys, bearer tokens, passwords, and secrets from logging strings.
    """
    if not text:
        return text

    masked = text
    for regex, replacement in SENSITIVE_PATTERNS:
        masked = regex.sub(replacement, masked)

    return masked
