from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.schemas.ask import AskRequest, AskResponse
from app.services.rag_service import RAGService

router = APIRouter()


@router.post(
    "",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Question Answering via RAG",
    description=(
        "Executes a complete Retrieval-Augmented Generation pipeline: "
        "processes user question, retrieves evidence chunks from the vector database, "
        "constructs context with strict grounding rules, prompts the LLM, and formats the response with source citations."
    ),
    responses={
        200: {"description": "Successfully synthesized answer with source citations"},
        422: {"description": "Malformed or invalid question"},
        502: {"description": "Upstream LLM service failure or timeout"},
    },
)
async def ask_question(
    request: AskRequest,
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
) -> AskResponse:
    """
    RAG Question-Answering endpoint.
    Retrieves evidence from ingested documents and generates grounded answers.
    """
    request_id = getattr(raw_request.state, "request_id", None)
    rag_service = RAGService(session=db)
    return await rag_service.answer_question(
        question=request.question,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        filters=request.filters,
        request_id=request_id,
    )
