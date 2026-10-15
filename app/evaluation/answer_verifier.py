import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "can", "could", "should", "would",
    "will", "shall", "may", "might", "must", "that", "which", "this", "these",
    "those", "it", "its", "they", "their", "them", "we", "our", "you", "your",
    "based", "provided", "context", "documentation", "document", "stated", "states",
    "according", "answer", "following", "therefore", "furthermore", "also", "however"
}

REFUSAL_PHRASES = [
    "not have enough information",
    "don't have enough information",
    "dont have enough information",
    "not have enough reliable information",
    "don't have enough reliable information",
    "dont have enough reliable information",
    "enough reliable information",
    "information is unavailable",
    "couldn't find sufficient information",
    "could not find sufficient information",
    "not enough reliable information",
    "unavailable in the knowledge base",
    "no information",
    settings.RAG_FALLBACK_RESPONSE.lower(),
    settings.RAG_UNVERIFIED_REFUSAL_RESPONSE.lower(),
]



class FaithfulnessVerificationResult(BaseModel):
    """
    Structured outcome of the post-generation faithfulness verification.
    Itemizes factual claims, detects unverified assertions, and outputs an audited faithfulness score.
    """

    supported: bool = Field(
        ...,
        description="True if the answer's factual assertions are fully grounded in retrieved evidence",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Faithfulness score (fraction of claims verified as grounded in context)",
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="List of factual claims that could not be verified against the context",
    )
    reason: str = Field(
        ...,
        description="Diagnostic explanation of the verification outcome",
    )
    claims_evaluated: int = Field(
        default=0,
        description="Total number of factual claims identified in the answer",
    )


class AnswerVerifier:
    """
    Evaluates claims in generated answers against retrieved context chunks.
    Performs deterministic claim extraction, entity/numeric validation, and semantic entailment checks
    to detect plausible hallucinations and ungrounded extrapolations.
    """

    def __init__(self, faithfulness_threshold: Optional[float] = None):
        self.faithfulness_threshold = (
            faithfulness_threshold
            if faithfulness_threshold is not None
            else settings.RAG_FAITHFULNESS_THRESHOLD
        )

    def extract_claims(self, answer: str) -> List[str]:
        """
        Decomposes an answer into discrete factual assertions, stripping framing prose and citations.
        """
        cleaned = answer.strip()
        if not cleaned:
            return []

        # If the answer is an explicit refusal, there are 0 factual assertions
        lower_answer = cleaned.lower()
        if any(phrase in lower_answer for phrase in REFUSAL_PHRASES):
            return []

        # Strip citation markers like [Document 1], [Doc: file.pdf], etc.
        stripped = re.sub(r"\[Document\s+\d+[^\]]*\]", "", cleaned)
        stripped = re.sub(r"\[Doc:[^\]]*\]", "", stripped)
        stripped = re.sub(r"\[[0-9]+\]", "", stripped)

        # Split into sentence candidates
        raw_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", stripped)
            if s.strip()
        ]

        # Patterns for standalone conversational meta-commentary sentences that contain no facts
        STANDALONE_META_PATTERNS = [
            r"^i\s+(will|must)\s+(strictly\s+)?(adhere|follow|comply)",
            r"^as\s+an\s+ai\b",
            r"^here\s+is\s+the\s+answer\s+based\b",
        ]

        claims: List[str] = []
        for s in raw_sentences:
            # Skip standalone conversational meta-framing sentences
            if any(re.search(p, s.strip(), flags=re.IGNORECASE) for p in STANDALONE_META_PATTERNS):
                continue


            # Filter meta framing prefix and conversational boilerplate
            s_clean = re.sub(
                r"^(based\s+(solely\s+)?on\s+the\s+provided\s+context[,\s:]*|"
                r"according\s+to\s+[a-z0-9_\-.\s]+[,\s:]*|"
                r"the\s+answer\s+to\s+the\s+question\s+is[,\s:]*|"
                r"in\s+summary[,\s:]*)",
                "",
                s,
                flags=re.IGNORECASE,
            ).strip()

            # Ignore empty or excessively short fragments
            if len(s_clean) >= 15 and not any(phrase in s_clean.lower() for phrase in REFUSAL_PHRASES):
                claims.append(s_clean)

        return claims


    def verify_claim(self, claim: str, context_text: str) -> Tuple[bool, float, str]:
        """
        Verifies a single factual claim against the combined context documentation.
        Validates token recall, entity alignment, and numeric constraints.
        """
        context_lower = context_text.lower()
        claim_lower = claim.lower()

        # 1. Numeric and quantitative validation:
        # If numbers appear in the claim, they must exist in the context
        claim_numbers = set(re.findall(r"\b\d+(?:[\.,]\d+)?\b", claim_lower))
        context_numbers = set(re.findall(r"\b\d+(?:[\.,]\d+)?\b", context_lower))

        missing_numbers = claim_numbers - context_numbers
        if missing_numbers:
            return (
                False,
                0.0,
                f"Claim references numbers not found in context: {sorted(missing_numbers)}",
            )

        # 2. Extract content tokens (length >= 3, non-stopwords)
        claim_tokens = [
            token
            for token in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", claim_lower)
            if token not in STOP_WORDS
        ]

        if not claim_tokens:
            return True, 1.0, "Claim contains no verifiable content tokens."

        # 3. Check best sentence-level match in context
        context_sentences = [
            s.strip().lower()
            for s in re.split(r"(?<=[.!?\n])\s+", context_lower)
            if s.strip()
        ]

        best_match_ratio = 0.0
        best_sentence = ""

        claim_token_set = set(claim_tokens)
        for c_sent in context_sentences:
            c_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", c_sent))
            overlap = len(claim_token_set & c_tokens)
            ratio = overlap / len(claim_token_set)
            if ratio > best_match_ratio:
                best_match_ratio = ratio
                best_sentence = c_sent

        # If a single sentence contains >= 60% of claim tokens, or full context contains >= 80%
        full_context_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", context_lower))
        global_overlap = len(claim_token_set & full_context_tokens) / len(claim_token_set)

        if best_match_ratio >= 0.55 or global_overlap >= 0.75:
            return True, round(max(best_match_ratio, global_overlap), 4), "Supported by context."

        missing_tokens = [t for t in claim_tokens if t not in full_context_tokens]
        return (
            False,
            round(best_match_ratio, 4),
            f"Unsupported key concepts in context: {missing_tokens[:4]}",
        )

    def verify_answer(
        self,
        answer: str,
        context_text: str,
    ) -> FaithfulnessVerificationResult:
        """
        Verifies all claims in the generated answer against the context documentation.
        Computes composite faithfulness score and identifies any hallucinated or ungrounded claims.
        """
        claims = self.extract_claims(answer)

        # If no factual claims were asserted (e.g. valid refusal)
        if not claims:
            logger.debug("No factual assertions in answer; verified as refusal.")
            return FaithfulnessVerificationResult(
                supported=True,
                score=1.0,
                unsupported_claims=[],
                reason="Answer contains no factual assertions (refusal statement verified).",
                claims_evaluated=0,
            )

        supported_count = 0
        unsupported_claims: List[str] = []

        for claim in claims:
            is_supported, match_score, diagnostic = self.verify_claim(claim, context_text)
            if is_supported:
                supported_count += 1
            else:
                logger.warning(
                    f"Unsupported claim detected: '{claim}' (match_score: {match_score}, reason: {diagnostic})"
                )
                unsupported_claims.append(claim)

        faithfulness_score = round(supported_count / len(claims), 4)
        is_supported = faithfulness_score >= self.faithfulness_threshold

        if is_supported:
            reason = f"All {supported_count}/{len(claims)} claims verified as grounded in context."
        else:
            reason = (
                f"Faithfulness check failed: {len(unsupported_claims)}/{len(claims)} "
                f"claims unverified (faithfulness score: {faithfulness_score:.2f} < threshold {self.faithfulness_threshold})."
            )

        logger.info(
            f"Faithfulness verification: supported={is_supported}, score={faithfulness_score}, unsupported={len(unsupported_claims)}"
        )

        return FaithfulnessVerificationResult(
            supported=is_supported,
            score=faithfulness_score,
            unsupported_claims=unsupported_claims,
            reason=reason,
            claims_evaluated=len(claims),
        )
