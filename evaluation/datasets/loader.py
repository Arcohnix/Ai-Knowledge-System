import json
from pathlib import Path
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.logging import get_logger
from app.ingestion.base import BaseEmbedder
from app.ingestion.chunker import SensibleTextChunker
from app.ingestion.cleaner import TextCleaner
from app.ingestion.parsers.parser_factory import ParserFactory
from app.models.document import DocumentChunkModel, DocumentModel, DocumentStatus
from evaluation.datasets.schemas import EvalDataset, EvalExample

logger = get_logger(__name__)

DEFAULT_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"


def load_eval_dataset(dataset_path: Optional[Path] = None) -> EvalDataset:
    """Loads and validates the benchmark evaluation dataset from JSON."""
    path = dataset_path or DEFAULT_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = EvalDataset.model_validate(data)
    logger.info(f"Loaded evaluation dataset '{dataset.name}' with {len(dataset.examples)} examples.")
    return dataset


# Canonical raw text fixtures for reproducibility in any environment
BENCHMARK_FIXTURES = [
    {
        "filename": "sample.txt",
        "title": "Engineering Operations Manual",
        "doc_type": "txt",
        "version": "1.0",
        "status": DocumentStatus.ACTIVE.value,
        "content": (
            "Engineering Operations Manual\n\n"
            "Section 1: Code Review Guidelines.\n"
            "All pull requests must have at least two peer approvals before merging.\n\n"
            "Section 2: Deployment Windows.\n"
            "Production deployments are permitted Tuesday through Thursday."
        ),
    },
    {
        "filename": "sample.md",
        "title": "Enterprise Security Architecture",
        "doc_type": "markdown",
        "version": "1.0",
        "status": DocumentStatus.ACTIVE.value,
        "content": (
            "# Enterprise Security Architecture\n\n"
            "## 1. Zero Trust Principles\n"
            "All incoming network traffic must authenticate via mutual TLS.\n\n"
            "## 2. Secrets Management\n"
            "No API keys or database credentials may be stored in source code."
        ),
    },
    {
        "filename": "sample.docx",
        "title": "Company Data Retention Policy",
        "doc_type": "docx",
        "version": "1.0",
        "status": DocumentStatus.ACTIVE.value,
        "content": (
            "Company Data Retention Policy\n\n"
            "Customer personal identification data must be retained for at least 7 years "
            "pursuant to regulatory requirements.\n\n"
            "Financial transaction ledgers must be archived in immutable storage."
        ),
    },
    {
        "filename": "sample.pdf",
        "title": "Annual Corporate Compliance Report",
        "doc_type": "pdf",
        "version": "1.0",
        "status": DocumentStatus.ACTIVE.value,
        "content": (
            "Annual Corporate Compliance Report\n\n"
            "All employees must complete mandatory ethics certification annually."
        ),
    },
    {
        "filename": "policy_2024.txt",
        "title": "Corporate Vacation Policy 2024",
        "doc_type": "txt",
        "version": "2024.1",
        "status": DocumentStatus.SUPERSEDED.value,
        "content": (
            "Corporate Vacation Policy (2024.1).\n"
            "Employees receive 20 annual vacation days upon hiring."
        ),
    },
    {
        "filename": "policy_2026.txt",
        "title": "Corporate Vacation Policy 2026",
        "doc_type": "txt",
        "version": "2026.1",
        "status": DocumentStatus.ACTIVE.value,
        "content": (
            "Corporate Vacation Policy (2026.1).\n"
            "Employees receive 25 annual vacation days upon hiring."
        ),
    },
    {
        "filename": "vacation_hr.txt",
        "title": "HR Vacation Guidelines",
        "doc_type": "txt",
        "version": "1.0",
        "status": DocumentStatus.ACTIVE.value,
        "content": (
            "HR Department Standard. All employees are entitled to 20 vacation days annually."
        ),
    },
    {
        "filename": "vacation_eng.txt",
        "title": "Engineering Vacation Guidelines",
        "doc_type": "txt",
        "version": "1.0",
        "status": DocumentStatus.ACTIVE.value,
        "content": (
            "Engineering Department Standard. All employees are entitled to 25 vacation days annually."
        ),
    },
]


async def seed_eval_documents_if_needed(
    session: AsyncSession,
    embedder: BaseEmbedder,
) -> int:
    """
    Ensures that all benchmark evaluation documents are ingested into the database session.
    If already present, skips ingestion. Returns the number of documents newly seeded.
    """
    cleaner = TextCleaner()
    chunker = SensibleTextChunker(chunk_size=300, chunk_overlap=30)

    seeded_count = 0
    for fixture in BENCHMARK_FIXTURES:
        # Check if document already exists
        query = select(DocumentModel).where(DocumentModel.filename == fixture["filename"])
        res = await session.execute(query)
        existing = res.scalar_one_or_none()
        if existing:
            continue

        # Ingest document
        cleaned_doc = cleaner.clean(fixture["content"])
        raw_chunks = chunker.split_text(cleaned_doc)

        chunk_texts = [c["content"] for c in raw_chunks]
        embeddings = await embedder.embed_documents(chunk_texts)

        vector_chunks = []
        for i, c in enumerate(raw_chunks):
            metadata = {
                "document_title": fixture["title"],
                "filename": fixture["filename"],
                "version": fixture["version"],
                "status": fixture["status"],
            }
            vector_chunks.append({
                "chunk_index": c["chunk_index"],
                "content": c["content"],
                "embedding": embeddings[i],
                "metadata": metadata,
            })

        import hashlib
        checksum = hashlib.sha256(fixture["content"].encode("utf-8")).hexdigest()

        doc_record = DocumentModel(
            filename=fixture["filename"],
            document_type=fixture["doc_type"],
            title=fixture["title"],
            source="evaluation_benchmark",
            version=fixture["version"],
            status=fixture["status"],
            checksum=checksum,
            doc_metadata={"benchmark": True},
        )
        session.add(doc_record)
        await session.flush()

        for vc in vector_chunks:
            chunk_record = DocumentChunkModel(
                document_id=doc_record.id,
                chunk_index=vc["chunk_index"],
                content=vc["content"],
                chunk_metadata=vc["metadata"],
                embedding=vc["embedding"],
            )
            session.add(chunk_record)

        await session.commit()
        seeded_count += 1
        logger.info(f"Seeded evaluation document: {fixture['filename']}")

    return seeded_count
