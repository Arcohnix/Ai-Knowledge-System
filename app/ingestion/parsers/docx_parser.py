import io
import os
from typing import BinaryIO, List, Union
from docx import Document
from app.core.exceptions import ValidationException
from app.ingestion.base import BaseDocumentParser, RawDocument
from app.ingestion.cleaner import TextCleaner


class DocxParser(BaseDocumentParser):
    """Parser for Microsoft Word documents (.docx) using python-docx."""

    @property
    def supported_mime_types(self) -> List[str]:
        return [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ]

    async def parse(self, content: Union[bytes, BinaryIO], filename: str) -> List[RawDocument]:
        if isinstance(content, bytes):
            stream = io.BytesIO(content)
        else:
            stream = content

        try:
            doc = Document(stream)
        except Exception as exc:
            raise ValidationException(f"Failed to parse DOCX document '{filename}': {exc}")

        paragraphs_text = []

        # Extract text from paragraphs
        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                paragraphs_text.append(text)

        # Extract text from tables
        for table in doc.tables:
            table_rows = []
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    table_rows.append(row_text)
            if table_rows:
                paragraphs_text.append("\n".join(table_rows))

        full_text = TextCleaner.clean("\n\n".join(paragraphs_text))

        # Core properties metadata
        title = ""
        author = None
        try:
            props = doc.core_properties
            title = props.title or ""
            author = props.author
        except Exception:
            pass

        if not title:
            title = os.path.splitext(filename)[0]

        metadata = {
            "source_type": "docx",
            "filename": filename,
            "title": title,
            "paragraph_count": len(doc.paragraphs),
            "table_count": len(doc.tables),
            "character_count": len(full_text),
            "author": author,
        }

        return [RawDocument(source=filename, content=full_text, metadata=metadata)]
