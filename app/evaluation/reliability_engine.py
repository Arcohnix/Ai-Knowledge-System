from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from app.core.config import settings
from app.core.logging import get_logger
from app.evaluation.answer_verifier import FaithfulnessVerificationResult

logger = get_logger(__name__)


class DecisionAction(str, Enum):
    """Operational action decided by the ReliabilityDecisionEngine."""

    ACCEPT = "ACCEPT"
    REGENERATE = "REGENERATE"
    REFUSE = "REFUSE"


class ReliabilityDecision(BaseModel):
    """
    Final decision and packaging of the answer verification audit.
    """

    action: DecisionAction = Field(..., description="Action taken: ACCEPT, REGENERATE, or REFUSE")
    final_answer: str = Field(..., description="Audited answer text (original or controlled refusal)")
    faithfulness_score: float = Field(..., ge=0.0, le=1.0)
    supported: bool = Field(...)
    unsupported_claims: List[str] = Field(default_factory=list)
    reason: str = Field(..., description="Rationale for decision")
    regeneration_attempt: int = Field(default=0)


class ReliabilityDecisionEngine:
    """
    Governs answer publication and hallucination mitigation.
    Decides whether to ACCEPT an answer, REGENERATE with stricter negative constraints,
    or REFUSE with a safe fallback response.
    """

    def __init__(
        self,
        max_regeneration_attempts: Optional[int] = None,
        enable_regeneration: Optional[bool] = None,
    ):
        self.max_regeneration_attempts = (
            max_regeneration_attempts
            if max_regeneration_attempts is not None
            else settings.RAG_MAX_REGENERATION_ATTEMPTS
        )
        self.enable_regeneration = (
            enable_regeneration
            if enable_regeneration is not None
            else settings.RAG_ENABLE_REGENERATION
        )

    def decide(
        self,
        verification: FaithfulnessVerificationResult,
        current_answer: str,
        regeneration_attempt: int = 0,
    ) -> ReliabilityDecision:
        """
        Evaluates the verification outcome and determines whether to ACCEPT, REGENERATE, or REFUSE.
        """
        # Case 1: Answer is fully faithful and supported
        if verification.supported:
            logger.info("ReliabilityDecisionEngine: ACCEPT (answer is fully supported).")
            return ReliabilityDecision(
                action=DecisionAction.ACCEPT,
                final_answer=current_answer.strip(),
                faithfulness_score=verification.score,
                supported=True,
                unsupported_claims=[],
                reason=verification.reason,
                regeneration_attempt=regeneration_attempt,
            )

        # Case 2: Answer has unsupported claims -> Evaluate if regeneration is permitted
        if self.enable_regeneration and regeneration_attempt < self.max_regeneration_attempts:
            logger.warning(
                f"ReliabilityDecisionEngine: REGENERATE (attempt {regeneration_attempt + 1}/{self.max_regeneration_attempts}). "
                f"Unsupported claims: {verification.unsupported_claims}"
            )
            return ReliabilityDecision(
                action=DecisionAction.REGENERATE,
                final_answer=current_answer.strip(),
                faithfulness_score=verification.score,
                supported=False,
                unsupported_claims=verification.unsupported_claims,
                reason=f"Regeneration triggered due to {len(verification.unsupported_claims)} unverified claims.",
                regeneration_attempt=regeneration_attempt + 1,
            )

        # Case 3: Unsupported and exhausted attempts -> REFUSE to prevent hallucination
        logger.warning(
            f"ReliabilityDecisionEngine: REFUSE (unverified claims detected: {verification.unsupported_claims})."
        )
        return ReliabilityDecision(
            action=DecisionAction.REFUSE,
            final_answer=settings.RAG_UNVERIFIED_REFUSAL_RESPONSE,
            faithfulness_score=verification.score,
            supported=False,
            unsupported_claims=verification.unsupported_claims,
            reason=f"Refused to publish answer with {len(verification.unsupported_claims)} unverified claims.",
            regeneration_attempt=regeneration_attempt,
        )

    def build_regeneration_prompt(
        self,
        unsupported_claims: List[str],
        question: str,
        context_text: str,
    ) -> str:
        """
        Builds a corrective prompt emphasizing the specific hallucinated/unverified claims to avoid.
        """
        claims_bullet = "\n".join(f"- {c}" for c in unsupported_claims)
        return (
            f"CRITICAL CORRECTION INSTRUCTION:\n"
            f"Your previous attempt generated the following UNVERIFIED claim(s) that are NOT supported by the context:\n"
            f"{claims_bullet}\n\n"
            f"Context Information:\n"
            f"---------------------\n"
            f"{context_text}\n"
            f"---------------------\n\n"
            f"User Question: {question}\n\n"
            f"Strict Rules for this retry:\n"
            f"1. Answer ONLY using statements directly confirmed in the context.\n"
            f"2. Do NOT mention or invent any of the unverified claims listed above.\n"
            f"3. If the context does not explicitly provide the answer, you must state:\n"
            f"   \"{settings.RAG_UNVERIFIED_REFUSAL_RESPONSE}\"\n\n"
            f"Answer:"
        )
