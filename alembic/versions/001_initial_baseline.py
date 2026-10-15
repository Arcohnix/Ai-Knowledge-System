"""initial_baseline

Revision ID: 001_initial_baseline
Revises: 
Create Date: 2026-10-15 18:20:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "001_initial_baseline"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Enable pgvector extension on PostgreSQL
    conn = op.get_bind()
    if conn.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2. Create documents table
    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("document_type", sa.String(length=50), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("source", sa.String(length=255), server_default="upload", nullable=False),
        sa.Column("version", sa.String(length=50), server_default="1.0", nullable=False),
        sa.Column("status", sa.String(length=50), server_default="ACTIVE", nullable=False),
        sa.Column("effective_date", sa.Date(), nullable=True),
        sa.Column("expiration_date", sa.Date(), nullable=True),
        sa.Column("checksum", sa.String(length=64), nullable=False),
        sa.Column("doc_metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("checksum"),
    )
    op.create_index(op.f("ix_documents_checksum"), "documents", ["checksum"], unique=True)
    op.create_index(op.f("ix_documents_document_type"), "documents", ["document_type"], unique=False)
    op.create_index(op.f("ix_documents_status"), "documents", ["status"], unique=False)
    op.create_index(op.f("ix_documents_effective_date"), "documents", ["effective_date"], unique=False)
    op.create_index(op.f("ix_documents_expiration_date"), "documents", ["expiration_date"], unique=False)

    # 3. Create document_chunks table
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("chunk_metadata", sa.JSON(), nullable=False),
        sa.Column("embedding", Vector(dim=384), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_document_chunks_document_id"), "document_chunks", ["document_id"], unique=False)

    if conn.dialect.name == "postgresql":
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw "
            "ON document_chunks USING hnsw (embedding vector_cosine_ops);"
        )

    # 4. Create query_traces table
    op.create_table(
        "query_traces",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("request_id", sa.String(length=100), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("cleaned_query", sa.Text(), nullable=False),
        sa.Column("top_k", sa.Integer(), server_default="4", nullable=False),
        sa.Column("similarity_threshold", sa.Float(), server_default="0.5", nullable=False),
        sa.Column("retrieval_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("top_similarity_score", sa.Float(), nullable=True),
        sa.Column("mean_similarity_score", sa.Float(), nullable=True),
        sa.Column("is_sufficient", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("sufficiency_reason", sa.String(length=100), nullable=True),
        sa.Column("contradiction_status", sa.String(length=50), nullable=True),
        sa.Column("generated_answer", sa.Text(), nullable=False),
        sa.Column("faithfulness_score", sa.Float(), nullable=True),
        sa.Column("supported", sa.Boolean(), nullable=True),
        sa.Column("unsupported_claims", sa.JSON(), nullable=False),
        sa.Column("final_decision", sa.String(length=50), nullable=False),
        sa.Column("refusal_reason", sa.String(length=100), nullable=True),
        sa.Column("citations", sa.JSON(), nullable=False),
        sa.Column("trace_details", sa.JSON(), nullable=False),
        sa.Column("latency_ms", sa.Float(), server_default="0.0", nullable=False),
        sa.Column("model_name", sa.String(length=100), server_default="default", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_query_traces_request_id"), "query_traces", ["request_id"], unique=False)
    op.create_index(op.f("ix_query_traces_final_decision"), "query_traces", ["final_decision"], unique=False)

    # 5. Create system_error_logs table
    op.create_table(
        "system_error_logs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("request_id", sa.String(length=100), nullable=True),
        sa.Column("endpoint", sa.String(length=255), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=False),
        sa.Column("stack_trace", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_system_error_logs_request_id"), "system_error_logs", ["request_id"], unique=False)


def downgrade() -> None:
    op.drop_table("system_error_logs")
    op.drop_table("query_traces")
    op.drop_table("document_chunks")
    op.drop_table("documents")
