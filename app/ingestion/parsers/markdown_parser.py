import os
import re
from typing import BinaryIO, List, Union
from app.ingestion.base import BaseDocumentParser, RawDocument
from app.ingestion.cleaner import TextCleaner


class MarkdownParser(BaseDocumentParser):
    """Parser for Markdown documents (.md, .markdown)."""

    @property
    def supported_mime_types(self) -> List[str]:
        return ["text/markdown", "text/x-markdown"]

    async def parse(self, content: Union[bytes, BinaryIO], filename: str) -> List[RawDocument]:
        if hasattr(content, "read"):
            data = content.read()
        else:
            data = content

        text = ""
        for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                text = data.decode(encoding)
                break
            except (UnicodeDecodeError, AttributeError):
                continue

        cleaned_text = TextCleaner.clean(text)

        # Extract title from markdown h1 or first non-empty line
        title = ""
        h1_match = re.search(r"^#\s+(.+)$", cleaned_text, flags=re.MULTILINE)
        if h1_match:
            title = h1_match.group(1).strip()
        else:
            first_line = next(
                (line.strip() for line in cleaned_text.splitlines() if line.strip()), ""
            )
            title = first_line[:100] if first_line else os.path.splitext(filename)[0]

        # Extract headings for document structure awareness
        headings = re.findall(r"^(#{1,6})\s+(.+)$", cleaned_text, flags=re.MULTILINE)
        heading_list = [{"level": len(h[0]), "title": h[1].strip()} for h in headings]

        metadata = {
            "source_type": "markdown",
            "filename": filename,
            "title": title,
            "character_count": len(cleaned_text),
            "line_count": len(cleaned_text.splitlines()),
            "headings": heading_list,
        }

        return [RawDocument(source=filename, content=cleaned_text, metadata=metadata)]
