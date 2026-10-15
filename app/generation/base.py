from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Encapsulates the standard response from any LLM provider."""

    content: str = Field(..., description="The generated text content")
    model_name: str = Field(..., description="The model name that generated the response")
    token_usage: Optional[Dict[str, int]] = Field(
        default=None, description="Prompt, completion, and total tokens"
    )
    raw_response: Optional[Dict[str, Any]] = Field(
        default=None, description="Raw provider-specific response dictionary"
    )


class BaseLLMProvider(ABC):
    """
    Abstract interface for LLM providers.
    Supports future interchangeable implementations:
    - Ollama / local models
    - OpenAI / Azure OpenAI / OpenAI-compatible
    - Anthropic Claude
    - Google Gemini
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the identifier of the provider (e.g. 'ollama', 'openai', 'mock')."""
        pass

    @property
    def model_name(self) -> str:
        """Returns the specific model name or identifier."""
        return getattr(self, "_model_name", "base-model")

    @abstractmethod
    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        """Synchronously generate a response for the given prompt and optional system prompt."""
        pass

    @abstractmethod
    async def agenerate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        """Asynchronously generate a response for the given prompt and optional system prompt."""
        pass
