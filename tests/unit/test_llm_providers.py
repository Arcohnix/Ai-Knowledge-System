import pytest
from app.core.exceptions import LLMGenerationError
from app.generation.factory import get_llm_provider
from app.generation.prompts import build_rag_user_prompt
from app.generation.providers.mock_provider import MockLLMProvider
from app.generation.providers.ollama_provider import OllamaLLMProvider


@pytest.mark.asyncio
async def test_mock_llm_grounded_answer():
    provider = MockLLMProvider()
    context = (
        "[Document 1] Title: Policy | File: policy.pdf | Chunk: 0\n"
        "Customer records must be retained for at least seven years.\n"
    )
    prompt = build_rag_user_prompt(
        question="How long must customer records be retained?",
        context=context,
    )
    response = await provider.agenerate(prompt)
    assert "[Document 1]" in response.content
    assert "retained" in response.content.lower()


@pytest.mark.asyncio
async def test_mock_llm_unavailable_answer():
    provider = MockLLMProvider()
    context = "No relevant documentation found."
    prompt = build_rag_user_prompt(
        question="What is the employee pet policy?",
        context=context,
    )
    response = await provider.agenerate(prompt)
    assert (
        "not have enough information" in response.content.lower()
        or "sufficient information" in response.content.lower()
    )


@pytest.mark.asyncio
async def test_mock_llm_failure():
    provider = MockLLMProvider(should_fail=True)
    with pytest.raises(LLMGenerationError, match="Mock LLM failure"):
        await provider.agenerate("Test question")


def test_llm_factory_selection():
    mock_p = get_llm_provider("mock")
    assert mock_p.provider_name == "mock"

    ollama_p = get_llm_provider("ollama")
    assert ollama_p.provider_name == "ollama"

    openai_p = get_llm_provider("openai")
    assert openai_p.provider_name == "openai"


@pytest.mark.asyncio
async def test_ollama_provider_connection_error():
    # Attempting to connect to an invalid port should raise LLMGenerationError cleanly
    provider = OllamaLLMProvider(base_url="http://127.0.0.1:54999", timeout=1.0)
    with pytest.raises(LLMGenerationError, match="Cannot connect to Ollama service"):
        await provider.agenerate("Hello")
