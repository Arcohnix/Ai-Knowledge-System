from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Iterator, Optional
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
    - OpenAI / Azure OpenAI
    - Anthropic Claude
    - Google Gemini
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the identifier of the provider (e.g. 'ollama', 'openai')."""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Synchronously generate a response for the given prompt."""
        pass

    @abstractmethod
    async def agenerate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Asynchronously generate a response for the given prompt."""
        pass
