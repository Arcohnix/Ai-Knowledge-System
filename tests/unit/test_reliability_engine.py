import pytest
from app.core.config import settings
from app.evaluation.answer_verifier import FaithfulnessVerificationResult
from app.evaluation.reliability_engine import (
    DecisionAction,
    ReliabilityDecisionEngine,
)


def test_reliability_engine_accept():
    engine = ReliabilityDecisionEngine()
    verification = FaithfulnessVerificationResult(
        supported=True,
        score=1.0,
        unsupported_claims=[],
        reason="All claims supported.",
        claims_evaluated=2,
    )
    decision = engine.decide(
        verification=verification,
        current_answer="All pull requests require two peer approvals.",
        regeneration_attempt=0,
    )
    assert decision.action == DecisionAction.ACCEPT
    assert decision.supported is True
    assert decision.faithfulness_score == 1.0
    assert decision.final_answer == "All pull requests require two peer approvals."
    assert decision.unsupported_claims == []


def test_reliability_engine_regenerate():
    engine = ReliabilityDecisionEngine(max_regeneration_attempts=1, enable_regeneration=True)
    verification = FaithfulnessVerificationResult(
        supported=False,
        score=0.5,
        unsupported_claims=["Employees receive a 5000 dollar bonus."],
        reason="1 claim unsupported.",
        claims_evaluated=2,
    )
    decision = engine.decide(
        verification=verification,
        current_answer="PRs need approval and employees receive a 5000 dollar bonus.",
        regeneration_attempt=0,
    )
    assert decision.action == DecisionAction.REGENERATE
    assert decision.supported is False
    assert decision.regeneration_attempt == 1
    assert "Employees receive a 5000 dollar bonus." in decision.unsupported_claims


def test_reliability_engine_refuse_after_exhausted_attempts():
    engine = ReliabilityDecisionEngine(max_regeneration_attempts=1, enable_regeneration=True)
    verification = FaithfulnessVerificationResult(
        supported=False,
        score=0.0,
        unsupported_claims=["Employees receive a 5000 dollar bonus."],
        reason="Faithfulness check failed.",
        claims_evaluated=1,
    )
    # Attempt 1 (already exhausted)
    decision = engine.decide(
        verification=verification,
        current_answer="Employees receive a 5000 dollar bonus.",
        regeneration_attempt=1,
    )
    assert decision.action == DecisionAction.REFUSE
    assert decision.supported is False
    assert decision.final_answer == settings.RAG_UNVERIFIED_REFUSAL_RESPONSE
    assert "Refused to publish answer" in decision.reason


def test_reliability_engine_refuse_when_regeneration_disabled():
    engine = ReliabilityDecisionEngine(enable_regeneration=False)
    verification = FaithfulnessVerificationResult(
        supported=False,
        score=0.0,
        unsupported_claims=["Arbitrary unverified fact."],
        reason="Faithfulness check failed.",
        claims_evaluated=1,
    )
    decision = engine.decide(
        verification=verification,
        current_answer="Arbitrary unverified fact.",
        regeneration_attempt=0,
    )
    assert decision.action == DecisionAction.REFUSE
    assert decision.final_answer == settings.RAG_UNVERIFIED_REFUSAL_RESPONSE


def test_build_regeneration_prompt():
    engine = ReliabilityDecisionEngine()
    prompt = engine.build_regeneration_prompt(
        unsupported_claims=["Claim A", "Claim B"],
        question="What is the policy?",
        context_text="Grounded context here.",
    )
    assert "CRITICAL CORRECTION INSTRUCTION" in prompt
    assert "- Claim A" in prompt
    assert "- Claim B" in prompt
    assert "Grounded context here." in prompt
    assert "User Question: What is the policy?" in prompt
    assert settings.RAG_UNVERIFIED_REFUSAL_RESPONSE in prompt
