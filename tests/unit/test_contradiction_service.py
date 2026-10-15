from datetime import date
import pytest
from app.retrieval.base import RetrievedChunk
from app.services.contradiction_service import (
    ConflictResolutionAction,
    ContradictionService,
    ContradictionStatus,
    compare_versions,
    parse_date,
    parse_version_tuple,
)


def test_compare_versions():
    assert compare_versions("2026.1", "2024.1") > 0
    assert compare_versions("2024.1", "2026.1") < 0
    assert compare_versions("2.0", "1.0") > 0
    assert compare_versions("v1.5", "1.5") == 0
    assert compare_versions("2.0.1", "2.0") > 0
    assert compare_versions("1.0", "1.0") == 0


def test_parse_date():
    assert parse_date("2026-01-15") == date(2026, 1, 15)
    assert parse_date(date(2025, 5, 1)) == date(2025, 5, 1)
    assert parse_date(None) is None
    assert parse_date("invalid-date") is None


def test_detect_numeric_conflict():
    service = ContradictionService()
    sent_a = "Employees receive 20 annual vacation days upon hiring."
    sent_b = "Employees receive 25 annual vacation days upon hiring."
    conflict = service.detect_pairwise_sentence_conflict(sent_a, sent_b)
    assert conflict is not None
    c_type, topic, desc = conflict
    assert c_type == "numeric_discrepancy"
    assert "20" in desc and "25" in desc


def test_detect_polarity_conflict():
    service = ContradictionService()
    sent_a = "Production deployments are permitted on Fridays with manager approval."
    sent_b = "Production deployments are strictly prohibited on Fridays under all circumstances."
    conflict = service.detect_pairwise_sentence_conflict(sent_a, sent_b)
    assert conflict is not None
    c_type, topic, desc = conflict
    assert c_type == "polarity_mismatch"


def test_concordant_sentences_agree():
    service = ContradictionService()
    sent_a = "Customer identification records must be retained for at least 7 years."
    sent_b = "According to retention standards, customer identification records must be retained for at least 7 years."
    conflict = service.detect_pairwise_sentence_conflict(sent_a, sent_b)
    assert conflict is None


def test_analyze_chunks_agreement():
    service = ContradictionService()
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Customer identification data must be retained for 7 years.",
            score=0.88,
            metadata={"filename": "retention_v1.docx", "version": "1.0", "status": "ACTIVE"},
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            document_id="doc-2",
            content="Financial ledgers and customer identification data must be retained for 7 years.",
            score=0.82,
            metadata={"filename": "finance_v1.docx", "version": "1.0", "status": "ACTIVE"},
        ),
    ]
    result = service.analyze_and_resolve(chunks)
    assert result.status == ContradictionStatus.AGREE
    assert not result.has_conflict
    assert result.resolution == ConflictResolutionAction.NO_CONFLICT
    assert len(result.resolved_chunks) == 2


def test_resolve_by_active_status():
    service = ContradictionService()
    chunk_old = RetrievedChunk(
        chunk_id="chunk-old",
        document_id="doc-old",
        content="Employees receive 20 annual vacation days.",
        score=0.91,  # Higher vector similarity score!
        metadata={"filename": "handbook_2022.pdf", "version": "2022.1", "status": "SUPERSEDED"},
    )
    chunk_new = RetrievedChunk(
        chunk_id="chunk-new",
        document_id="doc-new",
        content="Employees receive 25 annual vacation days.",
        score=0.84,
        metadata={"filename": "handbook_2026.pdf", "version": "2026.1", "status": "ACTIVE"},
    )
    result = service.analyze_and_resolve([chunk_old, chunk_new])
    assert result.status == ContradictionStatus.CONFLICT
    assert result.has_conflict is True
    assert result.is_resolved is True
    assert result.resolution == ConflictResolutionAction.RESOLVED_BY_ACTIVE_STATUS
    assert len(result.resolved_chunks) == 1
    assert result.resolved_chunks[0].chunk_id == "chunk-new"
    assert "handbook_2026.pdf" in result.rationale


def test_resolve_by_effective_date():
    service = ContradictionService(reference_date=date(2026, 6, 1))
    chunk_expired = RetrievedChunk(
        chunk_id="chunk-exp",
        document_id="doc-exp",
        content="Core working hours are 9:00 AM to 5:00 PM.",
        score=0.89,
        metadata={
            "filename": "work_policy_2023.txt",
            "version": "1.0",
            "status": "EXPIRED",
            "expiration_date": "2024-12-31",
        },
    )
    chunk_current = RetrievedChunk(
        chunk_id="chunk-curr",
        document_id="doc-curr",
        content="Core working hours are 10:00 AM to 4:00 PM.",
        score=0.85,
        metadata={
            "filename": "work_policy_2025.txt",
            "version": "2.0",
            "status": "ACTIVE",
            "effective_date": "2025-01-01",
        },
    )
    result = service.analyze_and_resolve([chunk_expired, chunk_current], ref_date=date(2026, 6, 1))
    assert result.has_conflict is True
    assert result.is_resolved is True
    assert len(result.resolved_chunks) == 1
    assert result.resolved_chunks[0].chunk_id == "chunk-curr"


def test_resolve_by_version_precedence():
    service = ContradictionService()
    chunk_v1 = RetrievedChunk(
        chunk_id="c-v1",
        document_id="doc-v1",
        content="All pull requests require 1 peer approval before merging.",
        score=0.87,
        metadata={"filename": "eng_manual.md", "version": "1.0", "status": "SUPERSEDED"},
    )
    chunk_v2 = RetrievedChunk(
        chunk_id="c-v2",
        document_id="doc-v2",
        content="All pull requests require 2 peer approvals before merging.",
        score=0.86,
        metadata={"filename": "eng_manual.md", "version": "2.0", "status": "ACTIVE"},
    )
    result = service.analyze_and_resolve([chunk_v1, chunk_v2])
    assert result.has_conflict is True
    assert result.is_resolved is True
    assert len(result.resolved_chunks) == 1
    assert result.resolved_chunks[0].chunk_id == "c-v2"


def test_unresolved_direct_conflict_between_active_documents():
    service = ContradictionService()
    chunk_dept_a = RetrievedChunk(
        chunk_id="c-a",
        document_id="doc-a",
        content="Employees receive 20 annual vacation days.",
        score=0.92,
        metadata={"filename": "hr_guide.docx", "version": "1.0", "status": "ACTIVE"},
    )
    chunk_dept_b = RetrievedChunk(
        chunk_id="c-b",
        document_id="doc-b",
        content="Employees receive 25 annual vacation days.",
        score=0.91,
        metadata={"filename": "engineering_benefits.docx", "version": "1.0", "status": "ACTIVE"},
    )
    result = service.analyze_and_resolve([chunk_dept_a, chunk_dept_b])
    assert result.status == ContradictionStatus.CONFLICT
    assert result.has_conflict is True
    assert result.is_resolved is False
    assert result.resolution == ConflictResolutionAction.UNRESOLVED
    assert len(result.conflicting_sources) == 2
    assert "hr_guide.docx" in result.conflicting_sources
    assert "engineering_benefits.docx" in result.conflicting_sources
