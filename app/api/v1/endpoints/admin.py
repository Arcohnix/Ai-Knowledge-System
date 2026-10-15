from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.services.admin_service import AdminService

router = APIRouter()


@router.get(
    "/overview",
    summary="Dashboard Overview KPIs",
    description="Returns top-level KPIs including document count, query volumes, acceptance rate, and recent errors.",
)
async def get_overview(db: AsyncSession = Depends(get_db)):
    admin_service = AdminService(session=db)
    return await admin_service.get_overview()


@router.get(
    "/queries",
    summary="Query Explorer",
    description="Returns paginated query traces with question text, decisions, faithfulness scores, and citations.",
)
async def get_queries(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    decision: Optional[str] = Query(None, description="Filter by decision: ACCEPTED, REGENERATED, REFUSED"),
    search: Optional[str] = Query(None, description="Search term in questions"),
    db: AsyncSession = Depends(get_db),
):
    admin_service = AdminService(session=db)
    return await admin_service.get_queries(limit=limit, offset=offset, decision=decision, search=search)


@router.get(
    "/queries/{request_id}",
    summary="End-to-End Query Trace",
    description="Returns the complete step-by-step audit trace from request to final decision for a specific query.",
)
async def get_query_trace(request_id: str, db: AsyncSession = Depends(get_db)):
    admin_service = AdminService(session=db)
    trace = await admin_service.get_query_trace(request_id=request_id)
    if not trace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query trace with request_id '{request_id}' not found.",
        )
    return trace


@router.get(
    "/retrieval",
    summary="Retrieval Analysis",
    description="Analyzes vector retrieval score distributions, threshold gating, and relevant vs. irrelevant ratios.",
)
async def get_retrieval_analysis(db: AsyncSession = Depends(get_db)):
    admin_service = AdminService(session=db)
    return await admin_service.get_retrieval_analysis()


@router.get(
    "/evaluation",
    summary="Evaluation Benchmarks History",
    description="Returns historical Ragas evaluation metrics over time and comparative experiment analysis.",
)
async def get_evaluation_history(db: AsyncSession = Depends(get_db)):
    admin_service = AdminService(session=db)
    return admin_service.get_evaluation_history()


@router.get(
    "/knowledge-base",
    summary="Knowledge Base Catalog",
    description="Returns comprehensive catalog of all documents, versions, lifecycle status, chunk counts, and embeddings.",
)
async def get_knowledge_base_status(db: AsyncSession = Depends(get_db)):
    admin_service = AdminService(session=db)
    return await admin_service.get_knowledge_base_status()


@router.get(
    "/reliability",
    summary="Reliability & Hallucination Metrics",
    description="Provides audit metrics on hallucination detection, ungrounded claims, insufficient evidence, and contradictions.",
)
async def get_reliability_metrics(db: AsyncSession = Depends(get_db)):
    admin_service = AdminService(session=db)
    return await admin_service.get_reliability_metrics()
