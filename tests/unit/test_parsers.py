import pytest
from app.core.exceptions import ValidationException
from app.ingestion.parsers import (
    DocxParser,
    MarkdownParser,
    ParserFactory,
    PdfParser,
    TxtParser,
)


@pytest.mark.asyncio
async def test_txt_parser():
    parser = TxtParser()
    content = b"Knowledge Base Policy\n\nRule 1: Always check sources."
    docs = await parser.parse(content, "policy.txt")
    assert len(docs) == 1
    assert "Rule 1: Always check sources." in docs[0].content
    assert docs[0].metadata["source_type"] == "txt"
    assert docs[0].metadata["filename"] == "policy.txt"


@pytest.mark.asyncio
async def test_markdown_parser():
    parser = MarkdownParser()
    content = b"# Security Playbook\n\n## Section A\nEnforce strict credentials."
    docs = await parser.parse(content, "playbook.md")
    assert len(docs) == 1
    assert "Enforce strict credentials." in docs[0].content
    assert docs[0].metadata["source_type"] == "markdown"
    assert len(docs[0].metadata["headings"]) == 2


@pytest.mark.asyncio
async def test_pdf_parser_with_sample():
    with open("tests/fixtures/sample.pdf", "rb") as f:
        pdf_bytes = f.read()

    parser = PdfParser()
    docs = await parser.parse(pdf_bytes, "sample.pdf")
    assert len(docs) == 1
    assert "Annual Corporate Compliance Report" in docs[0].content
    assert docs[0].metadata["total_pages"] == 1


@pytest.mark.asyncio
async def test_docx_parser_with_sample():
    with open("tests/fixtures/sample.docx", "rb") as f:
        docx_bytes = f.read()

    parser = DocxParser()
    docs = await parser.parse(docx_bytes, "sample.docx")
    assert len(docs) == 1
    assert "Customer personal identification data" in docs[0].content
    assert docs[0].metadata["source_type"] == "docx"


def test_parser_factory_resolution():
    assert isinstance(ParserFactory.get_parser("report.pdf"), PdfParser)
    assert isinstance(ParserFactory.get_parser("document.docx"), DocxParser)
    assert isinstance(ParserFactory.get_parser("notes.txt"), TxtParser)
    assert isinstance(ParserFactory.get_parser("guide.md"), MarkdownParser)
    assert isinstance(ParserFactory.get_parser("guide.markdown"), MarkdownParser)


def test_parser_factory_unsupported():
    with pytest.raises(ValidationException) as exc_info:
        ParserFactory.get_parser("archive.tar.gz")
    assert "No parser available" in str(exc_info.value)

    with pytest.raises(ValidationException):
        ParserFactory.get_document_type("script.py")
