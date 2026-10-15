from typing import Optional
from app.core.config import settings
from app.core.exceptions import ValidationException
from app.core.logging import get_logger
from app.generation.base import BaseLLMProvider
from app.generation.providers.mock_provider import MockLLMProvider
from app.generation.providers.ollama_provider import OllamaLLMProvider
from app.generation.providers.openai_provider import OpenAILLMProvider

logger = get_logger(__name__)


def get_llm_provider(
    provider_type: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> BaseLLMProvider:
    """
    Factory function returning an instantiated BaseLLMProvider based on configuration.
    Supports:
    - 'ollama': Local model served via Ollama API
    - 'openai' / 'api': Remote or local OpenAI-compatible API
    - 'mock': Deterministic mock provider for automated tests and offline CI/CD
    """
    selected_provider = (provider_type or settings.LLM_PROVIDER).lower()
    model = model_name or settings.LLM_MODEL

    logger.debug(f"Instantiating LLM provider: {selected_provider} with model: {model}")

    if selected_provider == "ollama":
        return OllamaLLMProvider(
            base_url=kwargs.get("base_url") or settings.OLLAMA_BASE_URL,
            model_name=model,
            temperature=kwargs.get("temperature", settings.LLM_TEMPERATURE),
            max_tokens=kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
            timeout=kwargs.get("timeout", settings.LLM_REQUEST_TIMEOUT),
        )
    elif selected_provider in ("openai", "api"):
        return OpenAILLMProvider(
            api_key=kwargs.get("api_key") or settings.OPENAI_API_KEY,
            base_url=kwargs.get("base_url") or settings.OPENAI_API_BASE_URL,
            model_name=model,
            temperature=kwargs.get("temperature", settings.LLM_TEMPERATURE),
            max_tokens=kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
            timeout=kwargs.get("timeout", settings.LLM_REQUEST_TIMEOUT),
        )
    elif selected_provider == "mock":
        return MockLLMProvider(
            model_name=model,
            custom_response=kwargs.get("custom_response"),
            should_fail=kwargs.get("should_fail", False),
        )
    else:
        raise ValidationException(
            f"Unsupported LLM provider: '{selected_provider}'. Supported providers are: 'ollama', 'openai', 'mock'."
        )
