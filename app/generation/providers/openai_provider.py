from typing import Any, Dict, Optional
import httpx
from app.core.config import settings
from app.core.exceptions import LLMGenerationError
from app.core.logging import get_logger
from app.generation.base import BaseLLMProvider, LLMResponse

logger = get_logger(__name__)


class OpenAILLMProvider(BaseLLMProvider):
    """
    LLM Provider integrating with OpenAI or OpenAI-compatible chat completion APIs.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        self._api_key = api_key or settings.OPENAI_API_KEY or "dummy-key"
        self._base_url = (base_url or settings.OPENAI_API_BASE_URL).rstrip("/")
        self._model_name = model_name or settings.LLM_MODEL
        self._temperature = (
            temperature if temperature is not None else settings.LLM_TEMPERATURE
        )
        self._max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        self._timeout = timeout or settings.LLM_REQUEST_TIMEOUT

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    def _build_payload(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return {
            "model": self._model_name,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        url = f"{self._base_url}/chat/completions"
        payload = self._build_payload(prompt, system_prompt)

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, json=payload, headers=self._headers())
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            logger.error(f"Cannot connect to OpenAI API at {self._base_url}: {exc}")
            raise LLMGenerationError(
                f"Cannot connect to OpenAI API at {self._base_url}.",
                details={"error": str(exc)},
            ) from exc
        except httpx.TimeoutException as exc:
            logger.error(f"OpenAI API request timed out after {self._timeout}s: {exc}")
            raise LLMGenerationError(
                f"OpenAI API request timed out after {self._timeout}s.",
                details={"timeout": self._timeout},
            ) from exc
        except Exception as exc:
            logger.error(f"OpenAI API generation failed: {exc}")
            raise LLMGenerationError(
                f"OpenAI API generation failed: {str(exc)}", details={"error": str(exc)}
            ) from exc

        choices = data.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = data.get("usage", {})
        token_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        return LLMResponse(
            content=content.strip(),
            model_name=self._model_name,
            token_usage=token_usage,
            raw_response=data,
        )

    async def agenerate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        url = f"{self._base_url}/chat/completions"
        payload = self._build_payload(prompt, system_prompt)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload, headers=self._headers())
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            logger.error(f"Cannot connect to OpenAI API at {self._base_url}: {exc}")
            raise LLMGenerationError(
                f"Cannot connect to OpenAI API at {self._base_url}.",
                details={"error": str(exc)},
            ) from exc
        except httpx.TimeoutException as exc:
            logger.error(f"OpenAI API request timed out after {self._timeout}s: {exc}")
            raise LLMGenerationError(
                f"OpenAI API request timed out after {self._timeout}s.",
                details={"timeout": self._timeout},
            ) from exc
        except Exception as exc:
            logger.error(f"OpenAI API generation failed: {exc}")
            raise LLMGenerationError(
                f"OpenAI API generation failed: {str(exc)}", details={"error": str(exc)}
            ) from exc

        choices = data.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = data.get("usage", {})
        token_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        return LLMResponse(
            content=content.strip(),
            model_name=self._model_name,
            token_usage=token_usage,
            raw_response=data,
        )
