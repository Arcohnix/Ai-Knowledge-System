from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.schemas.search import (
    DocumentSnippet,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from app.services.retrieval_service import RetrievalService

router = APIRouter()


@router.post(
    "",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic Vector Search",
    description="Embeds query and performs semantic similarity search across ingested document chunks using pgvector.",
    responses={
        200: {"description": "Retrieved semantically matching chunks with similarity scores"},
        422: {"description": "Invalid query parameters"},
    },
)
async def semantic_search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """
    Performs cosine similarity search using dense vector embeddings in PostgreSQL (pgvector).
    """
    service = RetrievalService(db)
    retrieved_chunks = await service.search(
        query=request.query,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        filters=request.filters,
    )

    results: list[SearchResultItem] = []
    for item in retrieved_chunks:
        doc_meta = item.metadata
        results.append(
            SearchResultItem(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                chunk_index=doc_meta.get("chunk_index", 0),
                content=item.content,
                similarity_score=item.score or 0.0,
                metadata=doc_meta,
                document=DocumentSnippet(
                    document_id=item.document_id,
                    filename=doc_meta.get("filename", "unknown"),
                    title=doc_meta.get("document_title", "Untitled"),
                    document_type=doc_meta.get("document_type", "unknown"),
                    source=doc_meta.get("source", "upload"),
                    version=doc_meta.get("version", "1.0"),
                ),
            )
        )

    return SearchResponse(
        query=request.query,
        total_results=len(results),
        results=results,
    )
