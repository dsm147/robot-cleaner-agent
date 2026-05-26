"""
pytest 共享配置：Mock 所有 LLM 调用，避免测试消耗 API 费用
"""
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def mock_llm():
    with patch("model.factory.chat_model") as mock:
        mock.invoke.return_value = MagicMock(
            content="这是 Mock 的回答",
            tool_calls=[],
            response_metadata={},
        )
        yield mock


@pytest.fixture
def mock_tool_response():
    def _make(content="", tool_calls=None):
        response = MagicMock()
        response.content = content
        response.tool_calls = tool_calls or []
        response.response_metadata = {}
        return response
    return _make


@pytest.fixture
def rag_service():
    from rag.rag_service import RagSummarizeService
    return RagSummarizeService(retriever_mode="vector")
