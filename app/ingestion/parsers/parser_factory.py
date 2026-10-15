import os
from typing import Dict, Optional
from app.core.exceptions import ValidationException
from app.ingestion.base import BaseDocumentParser
from app.ingestion.parsers.docx_parser import DocxParser
from app.ingestion.parsers.markdown_parser import MarkdownParser
from app.ingestion.parsers.pdf_parser import PdfParser
from app.ingestion.parsers.txt_parser import TxtParser


class ParserFactory:
    """Factory creating appropriate document parsers based on file extension and MIME type."""

    _PARSERS: Dict[str, BaseDocumentParser] = {
        "pdf": PdfParser(),
        "docx": DocxParser(),
        "txt": TxtParser(),
        "md": MarkdownParser(),
        "markdown": MarkdownParser(),
    }

    _MIME_MAP: Dict[str, str] = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/msword": "docx",
        "text/plain": "txt",
        "text/markdown": "md",
        "text/x-markdown": "md",
    }

    @classmethod
    def get_document_type(cls, filename: str, content_type: Optional[str] = None) -> str:
        """Determines normalized document type from filename extension or content_type."""
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if ext in cls._PARSERS:
            return "markdown" if ext in ["md", "markdown"] else ext

        if content_type and content_type in cls._MIME_MAP:
            mapped_ext = cls._MIME_MAP[content_type]
            return "markdown" if mapped_ext in ["md", "markdown"] else mapped_ext

        raise ValidationException(
            f"Unsupported file format '{filename}'. Supported formats are: PDF (.pdf), Word (.docx), Plain Text (.txt), Markdown (.md, .markdown)."
        )

    @classmethod
    def get_parser(cls, filename: str, content_type: Optional[str] = None) -> BaseDocumentParser:
        """Returns the parser instance for the given document."""
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if ext in cls._PARSERS:
            return cls._PARSERS[ext]

        if content_type and content_type in cls._MIME_MAP:
            mapped_ext = cls._MIME_MAP[content_type]
            return cls._PARSERS[mapped_ext]

        raise ValidationException(
            f"No parser available for '{filename}'. Supported formats are: PDF, DOCX, TXT, Markdown."
        )
