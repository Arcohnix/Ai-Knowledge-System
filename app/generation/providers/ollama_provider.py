import asyncio
import time
from typing import Any, Dict, Optional
import httpx
from app.core.config import settings
from app.core.exceptions import LLMGenerationError
from app.core.logging import get_logger
from app.generation.base import BaseLLMProvider, LLMResponse

logger = get_logger(__name__)


class OllamaLLMProvider(BaseLLMProvider):
    """
    LLM Provider integrating with a local Ollama server over HTTP.
    Uses the Ollama /api/chat endpoint with strict temperature and token controls.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        self._base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self._model_name = model_name or settings.LLM_MODEL
        self._temperature = (
            temperature if temperature is not None else settings.LLM_TEMPERATURE
        )
        self._max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        self._timeout = timeout or settings.LLM_REQUEST_TIMEOUT

    @property
    def provider_name(self) -> str:
        return "ollama"

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
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        url = f"{self._base_url}/api/chat"
        payload = self._build_payload(prompt, system_prompt)
        max_retries = kwargs.get("max_retries", settings.LLM_MAX_RETRIES)
        retry_delay = kwargs.get("retry_delay", settings.LLM_RETRY_DELAY)

        for attempt in range(max_retries + 1):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    break
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if attempt < max_retries:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Transient Ollama error ({exc}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Cannot connect to Ollama after {max_retries} retries: {exc}")
                    raise LLMGenerationError(
                        f"Cannot connect to Ollama service at {self._base_url} after {max_retries} retries: {exc}",
                        details={"error": str(exc), "base_url": self._base_url},
                    ) from exc
            except Exception as exc:
                logger.error(f"Ollama generation failed: {exc}")
                raise LLMGenerationError(
                    f"Ollama generation failed: {str(exc)}", details={"error": str(exc)}
                ) from exc

        content = data.get("message", {}).get("content", "")
        token_usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
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
        url = f"{self._base_url}/api/chat"
        payload = self._build_payload(prompt, system_prompt)
        max_retries = kwargs.get("max_retries", settings.LLM_MAX_RETRIES)
        retry_delay = kwargs.get("retry_delay", settings.LLM_RETRY_DELAY)

        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    break
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if attempt < max_retries:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Transient Ollama error ({exc}). Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Cannot connect to Ollama after {max_retries} retries: {exc}")
                    raise LLMGenerationError(
                        f"Cannot connect to Ollama service at {self._base_url} after {max_retries} retries: {exc}",
                        details={"error": str(exc), "base_url": self._base_url},
                    ) from exc
            except Exception as exc:
                logger.error(f"Ollama generation failed: {exc}")
                raise LLMGenerationError(
                    f"Ollama generation failed: {str(exc)}", details={"error": str(exc)}
                ) from exc

        content = data.get("message", {}).get("content", "")
        token_usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        }
        return LLMResponse(
            content=content.strip(),
            model_name=self._model_name,
            token_usage=token_usage,
            raw_response=data,
        )
