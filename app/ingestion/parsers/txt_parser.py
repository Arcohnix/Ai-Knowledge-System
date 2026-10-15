import os
from typing import BinaryIO, List, Union
from app.ingestion.base import BaseDocumentParser, RawDocument
from app.ingestion.cleaner import TextCleaner


class TxtParser(BaseDocumentParser):
    """Parser for plain text files (.txt)."""

    @property
    def supported_mime_types(self) -> List[str]:
        return ["text/plain"]

    async def parse(self, content: Union[bytes, BinaryIO], filename: str) -> List[RawDocument]:
        if hasattr(content, "read"):
            data = content.read()
        else:
            data = content

        # Decode text with fallback
        text = ""
        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                text = data.decode(encoding)
                break
            except (UnicodeDecodeError, AttributeError):
                continue

        cleaned_text = TextCleaner.clean(text)

        # Infer title from first non-empty line, or fallback to file basename
        first_line = next((line.strip() for line in cleaned_text.splitlines() if line.strip()), "")
        title = first_line[:100] if first_line else os.path.splitext(filename)[0]

        metadata = {
            "source_type": "txt",
            "filename": filename,
            "title": title,
            "character_count": len(cleaned_text),
            "line_count": len(cleaned_text.splitlines()),
        }

        return [RawDocument(source=filename, content=cleaned_text, metadata=metadata)]
