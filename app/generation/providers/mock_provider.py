import re
from typing import Any, Dict, Optional
from app.core.config import settings
from app.core.exceptions import LLMGenerationError
from app.generation.base import BaseLLMProvider, LLMResponse


class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic Mock LLM Provider for unit testing, CI/CD, and offline verification.
    Extracts relevant sentences from the context prompt or outputs the standard
    'information unavailable' refusal when context lacks evidence.
    """

    def __init__(
        self,
        model_name: str = "mock-llm-v1",
        custom_response: Optional[str] = None,
        should_fail: bool = False,
    ):
        self._model_name = model_name
        self.custom_response = custom_response
        self.should_fail = should_fail

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    def _generate_content(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        if self.should_fail:
            raise LLMGenerationError(
                "Mock LLM failure triggered for testing",
                details={"provider": "mock", "forced_failure": True},
            )

        if self.custom_response is not None:
            return self.custom_response

        # Extract context block and question from the user prompt
        context_match = re.search(
            r"Context Information:\s*-+\s*(.*?)\s*-+\s*User Question:\s*(.*?)(?:\n|$)",
            prompt,
            re.DOTALL,
        )
        if not context_match:
            # Fallback for plain prompt
            return settings.RAG_FALLBACK_RESPONSE

        context = context_match.group(1).strip()
        question = context_match.group(2).strip()

        if not context or "No relevant documentation found" in context:
            return settings.RAG_FALLBACK_RESPONSE

        # Filter generic words that appear in titles and questions
        generic_words = {
            "what", "when", "where", "which", "about", "rules", "does", "have",
            "policy", "company", "office", "standard", "guidelines", "document",
            "documentation", "bring", "bringing", "regarding", "require", "required",
            "should", "would", "could", "much", "many", "their", "there", "some",
            "into", "from", "with", "than", "that", "this", "also", "been", "being"
        }
        q_words = {
            w.lower()
            for w in re.findall(r"\b\w{3,}\b", question)
            if w.lower() not in generic_words
        }

        # Check if context documents contain question topic keywords
        matching_sentences = []
        doc_matches = re.findall(
            r"\[(Document\s+\d+)\][^\n]*\n(.*?)(?=(?:\[Document\s+\d+\]|\Z))",
            context,
            re.DOTALL,
        )

        min_overlap = 2 if len(q_words) >= 2 else 1

        if doc_matches and q_words:
            for doc_label, doc_text in doc_matches:
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc_text) if s.strip()]
                for sentence in sentences:
                    s_words = set(re.findall(r"\b\w{3,}\b", sentence.lower()))
                    overlap = q_words & s_words
                    if len(overlap) >= min_overlap:
                        matching_sentences.append((doc_label, sentence))

        if matching_sentences:
            evidence_parts = []
            for doc_label, sentence in matching_sentences[:3]:
                evidence_parts.append(f"{sentence} [{doc_label}]")
            return "Based on the provided documentation: " + " ".join(evidence_parts)

        # If context is present but contains no relevant information for question
        return settings.RAG_FALLBACK_RESPONSE

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        content = self._generate_content(prompt, system_prompt)
        return LLMResponse(
            content=content,
            model_name=self._model_name,
            token_usage={"prompt_tokens": len(prompt.split()), "completion_tokens": len(content.split()), "total_tokens": len(prompt.split()) + len(content.split())},
            raw_response={"mock": True},
        )

    async def agenerate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        content = self._generate_content(prompt, system_prompt)
        return LLMResponse(
            content=content,
            model_name=self._model_name,
            token_usage={"prompt_tokens": len(prompt.split()), "completion_tokens": len(content.split()), "total_tokens": len(prompt.split()) + len(content.split())},
            raw_response={"mock": True},
        )
