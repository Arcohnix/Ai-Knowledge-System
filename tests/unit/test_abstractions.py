import pytest
from app.generation.base import BaseLLMProvider, LLMResponse
from app.retrieval.base import BaseVectorStore, BaseRetriever, RetrievedChunk
from app.ingestion.base import BaseDocumentParser, BaseChunker, BaseEmbedder, RawDocument
from app.evaluation.base import BaseEvaluator, RAGEvaluationReport


def test_llm_provider_contract():
    """Verify BaseLLMProvider cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseLLMProvider()

    class MockLLM(BaseLLMProvider):
        @property
        def provider_name(self) -> str:
            return "mock"

        def generate(self, prompt: str, **kwargs) -> LLMResponse:
            return LLMResponse(content="test answer", model_name="mock-model")

        async def agenerate(self, prompt: str, **kwargs) -> LLMResponse:
            return LLMResponse(content="test answer", model_name="mock-model")

    mock_llm = MockLLM()
    assert mock_llm.provider_name == "mock"
    res = mock_llm.generate("hello")
    assert res.content == "test answer"
    assert res.model_name == "mock-model"


def test_vector_store_contract():
    """Verify BaseVectorStore cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseVectorStore()


def test_document_parser_contract():
    """Verify BaseDocumentParser cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseDocumentParser()


def test_chunker_contract():
    """Verify BaseChunker cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseChunker()


def test_embedder_contract():
    """Verify BaseEmbedder cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseEmbedder()


def test_evaluator_contract():
    """Verify BaseEvaluator cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseEvaluator()
