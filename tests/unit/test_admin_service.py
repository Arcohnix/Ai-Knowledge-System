import pytest
from app.models.document import DocumentChunkModel, DocumentModel
from app.models.trace import QueryTraceModel, SystemErrorLogModel
from app.services.admin_service import AdminService


@pytest.mark.asyncio
async def test_admin_service_overview_empty(test_db_session):
    admin_service = AdminService(session=test_db_session)
    overview = await admin_service.get_overview()

    assert overview["total_documents"] == 0
    assert overview["total_chunks"] == 0
    assert overview["total_questions"] == 0
    assert overview["acceptance_rate"] == 1.0
    assert overview["refusal_rate"] == 0.0
    assert overview["average_retrieval_score"] == 0.0
    assert overview["recent_system_errors"] == []


@pytest.mark.asyncio
async def test_admin_service_overview_with_data(test_db_session):
    admin_service = AdminService(session=test_db_session)

    # Insert sample document and chunk
    doc = DocumentModel(
        filename="test.txt",
        title="Test Document",
        document_type="txt",
        checksum="hash123",
        status="ACTIVE",
        version="1.0",
        doc_metadata={},
    )
    test_db_session.add(doc)
    await test_db_session.flush()

    chunk = DocumentChunkModel(
        document_id=doc.id,
        chunk_index=0,
        content="Sample content for unit test",
        embedding=[0.1] * 384,
    )
    test_db_session.add(chunk)

    # Insert sample traces: 1 ACCEPTED, 1 REFUSED
    trace1 = QueryTraceModel(
        request_id="req-1",
        question="What is the test policy?",
        cleaned_query="What is the test policy?",
        top_k=4,
        similarity_threshold=0.5,
        retrieval_count=2,
        top_similarity_score=0.90,
        mean_similarity_score=0.85,
        is_sufficient=True,
        final_decision="ACCEPTED",
        faithfulness_score=1.0,
        citations=[{"document": "test.txt", "chunk_id": str(chunk.id), "score": 0.90}],
        trace_details={"pipeline_stages": [{"stage": "test"}]},
        generated_answer="The test policy is valid.",
    )
    trace2 = QueryTraceModel(
        request_id="req-2",
        question="Unknown query?",
        cleaned_query="Unknown query?",
        top_k=4,
        similarity_threshold=0.5,
        retrieval_count=0,
        top_similarity_score=None,
        is_sufficient=False,
        final_decision="REFUSED",
        refusal_reason="INSUFFICIENT_EVIDENCE",
        citations=[],
        trace_details={"pipeline_stages": []},
        generated_answer="I couldn't find sufficient information.",
    )
    test_db_session.add_all([trace1, trace2])

    # Insert sample system error
    error_log = SystemErrorLogModel(
        request_id="req-err",
        endpoint="/api/v1/test",
        error_message="Test timeout occurred",
    )
    test_db_session.add(error_log)
    await test_db_session.commit()

    overview = await admin_service.get_overview()

    assert overview["total_documents"] == 1
    assert overview["total_chunks"] == 1
    assert overview["total_questions"] == 2
    assert overview["accepted_questions"] == 1
    assert overview["refused_questions"] == 1
    assert overview["acceptance_rate"] == 0.5
    assert overview["refusal_rate"] == 0.5
    assert overview["average_retrieval_score"] == 0.90
    assert len(overview["recent_system_errors"]) == 1
    assert overview["recent_system_errors"][0]["error_message"] == "Test timeout occurred"


@pytest.mark.asyncio
async def test_admin_service_get_queries_filtering_and_pagination(test_db_session):
    admin_service = AdminService(session=test_db_session)

    t1 = QueryTraceModel(
        request_id="q-1",
        question="Vacation policy question",
        cleaned_query="vacation policy question",
        final_decision="ACCEPTED",
        generated_answer="Answer 1",
        citations=[],
        trace_details={},
    )
    t2 = QueryTraceModel(
        request_id="q-2",
        question="Salary policy question",
        cleaned_query="salary policy question",
        final_decision="REFUSED",
        generated_answer="Answer 2",
        citations=[],
        trace_details={},
    )
    test_db_session.add_all([t1, t2])
    await test_db_session.commit()

    # Query all
    res = await admin_service.get_queries(limit=10, offset=0)
    assert res["total"] == 2
    assert len(res["items"]) == 2

    # Filter by decision
    res_accepted = await admin_service.get_queries(decision="ACCEPTED")
    assert res_accepted["total"] == 1
    assert res_accepted["items"][0]["request_id"] == "q-1"

    # Filter by search
    res_search = await admin_service.get_queries(search="Salary")
    assert res_search["total"] == 1
    assert res_search["items"][0]["request_id"] == "q-2"


@pytest.mark.asyncio
async def test_admin_service_get_query_trace(test_db_session):
    admin_service = AdminService(session=test_db_session)

    trace = QueryTraceModel(
        request_id="trace-unique-id",
        question="Explain zero trust",
        cleaned_query="explain zero trust",
        top_k=3,
        similarity_threshold=0.55,
        retrieval_count=1,
        top_similarity_score=0.88,
        is_sufficient=True,
        contradiction_status="AGREE",
        final_decision="ACCEPTED",
        faithfulness_score=1.0,
        supported=True,
        citations=[{"document": "security.md", "score": 0.88}],
        trace_details={
            "pipeline_stages": [
                {"stage": "query_processing"},
                {"stage": "retrieval"},
            ]
        },
        latency_ms=85.4,
        model_name="test-llama3",
        generated_answer="Zero trust requires mutual TLS.",
    )
    test_db_session.add(trace)
    await test_db_session.commit()

    # Found
    result = await admin_service.get_query_trace("trace-unique-id")
    assert result is not None
    assert result["request_id"] == "trace-unique-id"
    assert result["model_name"] == "test-llama3"
    assert len(result["pipeline_stages"]) == 2

    # Not found
    missing = await admin_service.get_query_trace("non-existent-id")
    assert missing is None


@pytest.mark.asyncio
async def test_admin_service_retrieval_analysis(test_db_session):
    admin_service = AdminService(session=test_db_session)

    trace = QueryTraceModel(
        request_id="ret-1",
        question="Retrieval analysis check",
        cleaned_query="retrieval analysis check",
        retrieval_count=2,
        top_similarity_score=0.78,
        is_sufficient=True,
        final_decision="ACCEPTED",
        citations=[{"document": "handbook.pdf", "score": 0.78}],
        trace_details={},
        generated_answer="Ok",
    )
    test_db_session.add(trace)
    await test_db_session.commit()

    analysis = await admin_service.get_retrieval_analysis()
    assert analysis["total_retrieval_events"] == 1
    assert analysis["score_distribution"]["0.70-0.85"] == 1
    assert len(analysis["top_retrieved_documents"]) == 1
    assert analysis["top_retrieved_documents"][0]["document"] == "handbook.pdf"


@pytest.mark.asyncio
async def test_admin_service_reliability_metrics(test_db_session):
    admin_service = AdminService(session=test_db_session)

    # 1 contradiction trace
    t_contra = QueryTraceModel(
        request_id="rel-contra",
        question="Conflicting vacation days",
        cleaned_query="conflicting vacation days",
        is_sufficient=True,
        contradiction_status="CONFLICT",
        final_decision="REFUSED",
        refusal_reason="CONFLICTING_EVIDENCE",
        citations=[],
        trace_details={},
        generated_answer="Conflicting documents.",
    )
    # 1 ungrounded hallucination trace
    t_hallu = QueryTraceModel(
        request_id="rel-hallu",
        question="Hallucination test",
        cleaned_query="hallucination test",
        is_sufficient=True,
        final_decision="REFUSED",
        refusal_reason="UNVERIFIED_CLAIMS",
        unsupported_claims=["Company offers free spacecraft tickets."],
        citations=[],
        trace_details={},
        generated_answer="We offer spacecraft tickets.",
    )
    test_db_session.add_all([t_contra, t_hallu])
    await test_db_session.commit()

    rel = await admin_service.get_reliability_metrics()
    assert rel["total_queries_evaluated"] == 2
    assert rel["contradiction_cases"] == 1
    assert rel["hallucinations_detected_count"] == 1
    assert rel["unsupported_claims_log"][0]["claim"] == "Company offers free spacecraft tickets."
    assert rel["refusal_reasons_breakdown"]["CONFLICTING_EVIDENCE"] == 1
    assert rel["refusal_reasons_breakdown"]["UNVERIFIED_CLAIMS"] == 1


@pytest.mark.asyncio
async def test_admin_service_knowledge_base_status(test_db_session):
    admin_service = AdminService(session=test_db_session)

    doc = DocumentModel(
        filename="compliance.docx",
        title="Compliance 2026",
        document_type="docx",
        version="2026.1",
        status="ACTIVE",
        checksum="hash-compliance",
        doc_metadata={},
    )
    test_db_session.add(doc)
    await test_db_session.flush()

    chunk1 = DocumentChunkModel(
        document_id=doc.id,
        chunk_index=0,
        content="Chunk 1",
        embedding=[0.05] * 384,
    )
    test_db_session.add(chunk1)
    await test_db_session.commit()

    kb = await admin_service.get_knowledge_base_status()
    assert kb["total_documents"] == 1
    assert kb["documents"][0]["filename"] == "compliance.docx"
    assert kb["documents"][0]["chunk_count"] == 1
    assert kb["documents"][0]["embedding_status"] == "EMBEDDED"
