import io
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_ingest_txt_document(async_client: AsyncClient):
    with open("tests/fixtures/sample.txt", "rb") as f:
        content = f.read()

    files = {"file": ("operations.txt", content, "text/plain")}
    response = await async_client.post("/api/v1/documents", files=files)

    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "operations.txt"
    assert data["document_type"] == "txt"
    assert "Engineering Operations Manual" in data["title"]
    assert data["chunk_count"] > 0
    assert "checksum" in data
    assert "metadata" in data
    assert data["metadata"]["source_type"] == "txt"


@pytest.mark.asyncio
async def test_ingest_markdown_document(async_client: AsyncClient):
    with open("tests/fixtures/sample.md", "rb") as f:
        content = f.read()

    files = {"file": ("security.md", content, "text/markdown")}
    response = await async_client.post("/api/v1/documents", files=files)

    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "security.md"
    assert data["document_type"] == "markdown"
    assert data["title"] == "Enterprise Security Architecture"
    assert data["chunk_count"] > 0
    assert len(data["metadata"]["headings"]) >= 2


@pytest.mark.asyncio
async def test_ingest_docx_document(async_client: AsyncClient):
    with open("tests/fixtures/sample.docx", "rb") as f:
        content = f.read()

    files = {
        "file": (
            "retention.docx",
            content,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    }
    response = await async_client.post("/api/v1/documents", files=files)

    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "retention.docx"
    assert data["document_type"] == "docx"
    assert data["chunk_count"] > 0


@pytest.mark.asyncio
async def test_ingest_pdf_document(async_client: AsyncClient):
    with open("tests/fixtures/sample.pdf", "rb") as f:
        content = f.read()

    files = {"file": ("compliance.pdf", content, "application/pdf")}
    response = await async_client.post("/api/v1/documents", files=files)

    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "compliance.pdf"
    assert data["document_type"] == "pdf"
    assert data["chunk_count"] > 0
    assert data["metadata"]["total_pages"] == 1


@pytest.mark.asyncio
async def test_duplicate_document_rejection(async_client: AsyncClient):
    content = b"Unique document content for testing duplicate detection.\n\nSection 1."
    files = {"file": ("unique_doc.txt", content, "text/plain")}

    # First upload: 201 Created
    res1 = await async_client.post("/api/v1/documents", files=files)
    assert res1.status_code == 201
    doc_id = res1.json()["document_id"]

    # Second upload of identical content: 409 Conflict
    files2 = {"file": ("unique_doc_duplicate.txt", content, "text/plain")}
    res2 = await async_client.post("/api/v1/documents", files=files2)
    assert res2.status_code == 409
    err = res2.json()["error"]
    assert err["code"] == "CONFLICT"
    assert err["details"]["existing_document_id"] == doc_id


@pytest.mark.asyncio
async def test_invalid_file_type_rejection(async_client: AsyncClient):
    files = {"file": ("script.invalid", b"binary content", "application/octet-stream")}
    response = await async_client.post("/api/v1/documents", files=files)

    assert response.status_code == 422
    err = response.json()["error"]
    assert err["code"] == "VALIDATION_ERROR"
    assert "Unsupported file format" in err["message"]


@pytest.mark.asyncio
async def test_empty_file_rejection(async_client: AsyncClient):
    files = {"file": ("empty.txt", b"", "text/plain")}
    response = await async_client.post("/api/v1/documents", files=files)

    assert response.status_code == 422
    err = response.json()["error"]
    assert err["code"] == "VALIDATION_ERROR"
    assert "empty" in err["message"]


@pytest.mark.asyncio
async def test_get_and_list_documents(async_client: AsyncClient):
    # Upload sample to list
    content = b"Content for listing test.\n\nParagraph 1.\n\nParagraph 2."
    files = {"file": ("list_test.txt", content, "text/plain")}
    upload_res = await async_client.post("/api/v1/documents", files=files)
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    # List documents
    list_res = await async_client.get("/api/v1/documents?skip=0&limit=10")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] >= 1
    assert any(d["document_id"] == doc_id for d in list_data["documents"])

    # Filter by type
    filter_res = await async_client.get("/api/v1/documents?document_type=txt")
    assert filter_res.status_code == 200
    for doc in filter_res.json()["documents"]:
        assert doc["document_type"] == "txt"

    # Get single document with chunks
    detail_res = await async_client.get(f"/api/v1/documents/{doc_id}")
    assert detail_res.status_code == 200
    detail_data = detail_res.json()
    assert detail_data["document_id"] == doc_id
    assert len(detail_data["chunks"]) > 0

    chunk = detail_data["chunks"][0]
    assert "chunk_id" in chunk
    assert chunk["document_id"] == doc_id
    assert chunk["chunk_index"] == 0
    assert "content" in chunk
    assert "metadata" in chunk
    assert "token_estimate" in chunk["metadata"]


@pytest.mark.asyncio
async def test_delete_document_cascades(async_client: AsyncClient):
    content = b"Content to be deleted.\n\nWill be removed."
    files = {"file": ("to_delete.txt", content, "text/plain")}
    upload_res = await async_client.post("/api/v1/documents", files=files)
    doc_id = upload_res.json()["document_id"]

    # Delete
    del_res = await async_client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # Confirm it's gone
    get_res = await async_client.get(f"/api/v1/documents/{doc_id}")
    assert get_res.status_code == 404
