from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import Boolean, Float, Integer, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base
from app.models.base import TimestampMixin, UUIDMixin


class QueryTraceModel(Base, UUIDMixin, TimestampMixin):
    """
    SQLAlchemy model recording end-to-end multi-stage execution traces
    for every question answered or refused by the AI Knowledge System.
    """

    __tablename__ = "query_traces"

    request_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    cleaned_query: Mapped[str] = mapped_column(Text, nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False, default=4)
    similarity_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    # Retrieval Stage
    retrieval_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    top_similarity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mean_similarity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Pre-generation Evidence Evaluation Stage
    is_sufficient: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    sufficiency_reason: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Contradiction & Versioning Stage
    contradiction_status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Generation & Verification Stage
    generated_answer: Mapped[str] = mapped_column(Text, nullable=False)
    faithfulness_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    supported: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    unsupported_claims: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)

    # Final Decision Stage (ACCEPTED, REGENERATED, REFUSED)
    final_decision: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    refusal_reason: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Output Citations & Full Multi-Stage Diagnostic Trace
    citations: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    trace_details: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False, default="default")


class SystemErrorLogModel(Base, UUIDMixin, TimestampMixin):
    """
    SQLAlchemy model recording unhandled exceptions and system faults for operational visibility.
    """

    __tablename__ = "system_error_logs"

    request_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    endpoint: Mapped[str] = mapped_column(String(255), nullable=False)
    error_message: Mapped[str] = mapped_column(Text, nullable=False)
    stack_trace: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
