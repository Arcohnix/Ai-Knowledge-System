from datetime import date
from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.schemas.common import APIResponse
from app.schemas.document import (
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentResponse,
)
from app.services.ingestion_service import IngestionService

router = APIRouter()


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document",
    description="Validates, parses, cleans, chunks, and persists a document (PDF, DOCX, TXT, Markdown) with version lifecycle metadata.",
    responses={
        201: {"description": "Document parsed and indexed successfully"},
        400: {"description": "Invalid file format or empty payload"},
        409: {"description": "Duplicate document detected via SHA-256 checksum"},
        422: {"description": "Semantic validation or parsing failure"},
    },
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to ingest (PDF, DOCX, TXT, MD)"),
    version: str = Form("1.0", description="Document version identifier (e.g. '1.0', '2026.1')"),
    status_val: str = Form("ACTIVE", alias="status", description="Lifecycle status: ACTIVE, SUPERSEDED, or EXPIRED"),
    effective_date: Optional[date] = Form(None, description="Optional policy effective date (YYYY-MM-DD)"),
    expiration_date: Optional[date] = Form(None, description="Optional policy expiration date (YYYY-MM-DD)"),
    chunk_size: Optional[int] = Query(
        None, ge=50, le=4000, description="Optional override for chunk character length"
    ),
    chunk_overlap: Optional[int] = Query(
        None, ge=0, le=1000, description="Optional override for chunk overlap character length"
    ),
    source: str = Form("upload", description="Document source identifier"),
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """
    Ingests an uploaded document into structured text chunks with version and lifecycle status.
    """
    file_bytes = await file.read()
    service = IngestionService(db)

    doc = await service.ingest_document(
        filename=file.filename or "unknown",
        content=file_bytes,
        content_type=file.content_type,
        source=source,
        version=version,
        status=status_val.upper(),
        effective_date=effective_date,
        expiration_date=expiration_date,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    doc_response = DocumentResponse.model_validate(doc)
    doc_response.chunk_count = len(doc.chunks)
    return doc_response



@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List ingested documents",
    description="Returns a paginated list of ingested documents with optional type filtering.",
)
async def list_documents(
    skip: int = Query(0, ge=0, description="Records to skip"),
    limit: int = Query(20, ge=1, le=100, description="Maximum records to return"),
    document_type: Optional[str] = Query(
        None, description="Filter by type (e.g. 'pdf', 'docx', 'txt', 'markdown')"
    ),
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    service = IngestionService(db)
    docs, total = await service.list_documents(
        skip=skip, limit=limit, document_type=document_type
    )

    doc_responses = []
    for doc in docs:
        item = DocumentResponse.model_validate(doc)
        item.chunk_count = len(doc.chunks)
        doc_responses.append(item)

    return DocumentListResponse(
        total=total,
        skip=skip,
        limit=limit,
        documents=doc_responses,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details and chunks",
    description="Returns document metadata alongside all extracted text chunks.",
    responses={
        404: {"description": "Document not found"},
    },
)
async def get_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> DocumentDetailResponse:
    service = IngestionService(db)
    doc = await service.get_document(document_id)
    doc_detail = DocumentDetailResponse.model_validate(doc)
    doc_detail.chunk_count = len(doc.chunks)
    return doc_detail


@router.delete(
    "/{document_id}",
    response_model=APIResponse[dict],
    summary="Delete an ingested document",
    description="Removes the document and cascades deletion to all its chunks.",
    responses={
        404: {"description": "Document not found"},
    },
)
async def delete_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> APIResponse[dict]:
    service = IngestionService(db)
    await service.delete_document(document_id)
    return APIResponse(
        success=True,
        message=f"Document {document_id} and all related chunks were deleted successfully.",
        data={"document_id": str(document_id)},
    )
