import pytest
from app.evaluation.answer_verifier import AnswerVerifier


def test_answer_verifier_extract_claims():
    verifier = AnswerVerifier()
    answer = (
        "Based on the provided context, customer data must be retained for 7 years. "
        "Furthermore, according to document 1, production deployments run on Tuesdays. [Document 1]"
    )
    claims = verifier.extract_claims(answer)
    assert len(claims) == 2
    assert "customer data must be retained for 7 years" in claims[0].lower()
    assert "production deployments run on tuesdays" in claims[1].lower()


def test_answer_verifier_extract_claims_ignores_refusal():
    verifier = AnswerVerifier()
    answer = "I couldn't find sufficient information in the available knowledge base to answer this question reliably."
    claims = verifier.extract_claims(answer)
    assert claims == []


def test_answer_verifier_supported_claim():
    verifier = AnswerVerifier()
    context = (
        "Engineering Operations Manual. Production deployments are permitted Tuesday through Thursday. "
        "Emergency hotfixes require explicit VP approval."
    )
    claim = "Production deployments are permitted Tuesday through Thursday."
    supported, score, reason = verifier.verify_claim(claim, context)
    assert supported is True
    assert score >= 0.8
    assert "Supported by context" in reason


def test_answer_verifier_detects_numeric_hallucination():
    verifier = AnswerVerifier()
    context = (
        "Customer personal identification data must be retained for at least 7 years pursuant to regulatory requirements."
    )
    # Model hallucinated 3 years instead of 7 years
    hallucinated_claim = "Customer personal identification data must be retained for 3 years."
    supported, score, reason = verifier.verify_claim(hallucinated_claim, context)
    assert supported is False
    assert score == 0.0
    assert "Claim references numbers not found in context" in reason
    assert "3" in reason


def test_answer_verifier_detects_unsupported_concepts():
    verifier = AnswerVerifier()
    context = (
        "All pull requests must have at least two peer approvals before merging."
    )
    # Model hallucinated free gym memberships
    hallucinated_claim = "The company provides all employees with free gym memberships and wellness stipends."
    supported, score, reason = verifier.verify_claim(hallucinated_claim, context)
    assert supported is False
    assert "Unsupported key concepts in context" in reason


def test_answer_verifier_full_answer_verification_passes():
    verifier = AnswerVerifier()
    context = (
        "All pull requests must have at least two peer approvals before merging to main branch. "
        "Production deployments are permitted Tuesday through Thursday."
    )
    answer = (
        "Pull requests require two peer approvals before merging. "
        "Production deployments are allowed Tuesday through Thursday."
    )
    result = verifier.verify_answer(answer, context)
    assert result.supported is True
    assert result.score == 1.0
    assert len(result.unsupported_claims) == 0
    assert result.claims_evaluated == 2


def test_answer_verifier_full_answer_verification_fails_on_hallucination():
    verifier = AnswerVerifier()
    context = (
        "Customer personal identification data must be retained for at least 7 years."
    )
    # Answer contains 1 grounded claim and 1 completely hallucinated claim
    answer = (
        "Customer data must be retained for 7 years. "
        "In addition, employees receive a 10000 dollar annual bonus for compliance."
    )
    result = verifier.verify_answer(answer, context)
    assert result.supported is False
    assert result.score == 0.5  # 1 out of 2 claims supported
    assert len(result.unsupported_claims) == 1
    assert "10000 dollar annual bonus" in result.unsupported_claims[0]
    assert "Faithfulness check failed" in result.reason


def test_answer_verifier_refusal_answer():
    verifier = AnswerVerifier()
    context = "Some context about retention."
    answer = "I don't have enough reliable information to answer this."
    result = verifier.verify_answer(answer, context)
    assert result.supported is True
    assert result.score == 1.0
    assert result.claims_evaluated == 0
    assert "refusal statement verified" in result.reason
