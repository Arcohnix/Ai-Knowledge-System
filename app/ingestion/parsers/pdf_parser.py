import io
import os
from typing import BinaryIO, List, Union
from pypdf import PdfReader
from app.core.exceptions import ValidationException
from app.ingestion.base import BaseDocumentParser, RawDocument
from app.ingestion.cleaner import TextCleaner


class PdfParser(BaseDocumentParser):
    """Parser for Portable Document Format (.pdf) files using pypdf."""

    @property
    def supported_mime_types(self) -> List[str]:
        return ["application/pdf"]

    async def parse(self, content: Union[bytes, BinaryIO], filename: str) -> List[RawDocument]:
        if isinstance(content, bytes):
            stream = io.BytesIO(content)
        else:
            stream = content

        try:
            reader = PdfReader(stream)
        except Exception as exc:
            raise ValidationException(f"Failed to parse PDF document '{filename}': {exc}")

        if reader.is_encrypted:
            try:
                # Try decrypting with empty password
                reader.decrypt("")
            except Exception:
                raise ValidationException(
                    f"PDF document '{filename}' is password-protected and cannot be read."
                )

        total_pages = len(reader.pages)
        page_texts = []

        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_clean = TextCleaner.clean(page_text)
            if page_clean:
                page_texts.append(f"--- Page {page_idx + 1} ---\n{page_clean}")

        full_text = "\n\n".join(page_texts)

        # Extract PDF metadata if available
        pdf_info = reader.metadata or {}
        extracted_title = pdf_info.get("/Title")
        author = pdf_info.get("/Author")

        title = extracted_title if extracted_title else os.path.splitext(filename)[0]

        metadata = {
            "source_type": "pdf",
            "filename": filename,
            "title": title,
            "total_pages": total_pages,
            "character_count": len(full_text),
            "author": str(author) if author else None,
        }

        return [RawDocument(source=filename, content=full_text, metadata=metadata)]
