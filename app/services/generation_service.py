from typing import Optional
from app.core.exceptions import LLMGenerationError
from app.core.logging import get_logger
from app.generation.base import BaseLLMProvider, LLMResponse
from app.generation.factory import get_llm_provider
from app.generation.prompts import RAG_SYSTEM_PROMPT, build_rag_user_prompt
from app.services.context_service import ConstructedContext

logger = get_logger(__name__)


class LLMGenerationService:
    """
    Service responsible for orchestrating LLM prompt formatting and response generation.
    Interacts with the provider-agnostic BaseLLMProvider.
    """

    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self.provider = provider or get_llm_provider()

    async def generate_answer(
        self,
        question: str,
        context: ConstructedContext,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Assembles system instructions, context, and question, then calls the configured LLM provider.
        """
        sys_prompt = system_prompt or RAG_SYSTEM_PROMPT
        user_prompt = build_rag_user_prompt(
            question=question,
            context=context.context_text,
        )

        logger.info(
            f"Invoking LLM [{self.provider.provider_name}/{self.provider.model_name}] "
            f"for question: '{question[:60]}...'"
        )

        try:
            response = await self.provider.agenerate(
                prompt=user_prompt,
                system_prompt=sys_prompt,
            )
            logger.debug(
                f"LLM generated {len(response.content)} chars (tokens: {response.token_usage})"
            )
            return response
        except LLMGenerationError:
            raise
        except Exception as exc:
            logger.error(f"Unexpected generation error: {exc}")
            raise LLMGenerationError(
                f"Failed to generate answer from LLM: {str(exc)}",
                details={"provider": self.provider.provider_name, "error": str(exc)},
            ) from exc

    async def generate_correction(
        self,
        correction_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generates an answer using a corrective prompt with negative constraints
        aimed at recovering from unverified or hallucinated assertions.
        """
        sys_prompt = system_prompt or RAG_SYSTEM_PROMPT
        logger.info(
            f"Invoking LLM for corrective regeneration [{self.provider.provider_name}/{self.provider.model_name}]"
        )
        try:
            response = await self.provider.agenerate(
                prompt=correction_prompt,
                system_prompt=sys_prompt,
            )
            logger.debug(
                f"LLM generated correction {len(response.content)} chars (tokens: {response.token_usage})"
            )
            return response
        except LLMGenerationError:
            raise
        except Exception as exc:
            logger.error(f"Unexpected corrective generation error: {exc}")
            raise LLMGenerationError(
                f"Failed to generate corrective answer from LLM: {str(exc)}",
                details={"provider": self.provider.provider_name, "error": str(exc)},
            ) from exc

