"""
pytest 共享配置：Mock 所有 LLM 调用，避免测试消耗 API 费用
"""
import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def mock_llm():
    """Mock chat_model 使其返回固定的 AIMessage，避免 LangGraph 消息处理报错"""
    with patch("model.factory.chat_model", autospec=BaseChatModel) as mock:
        # LangChain create_agent 内部会调用 bind_tools
        # 让 bind_tools 返回自身，形成链式调用
        mock.bind_tools.return_value = mock
        mock.invoke.return_value = AIMessage(
            content="这是 Mock 的回答",
            tool_calls=[],
            response_metadata={},
        )
        yield mock


@pytest.fixture
def mock_tool_response():
    def _make(content="", tool_calls=None):
        from langchain_core.messages import AIMessage

        response = AIMessage(
            content=content,
            tool_calls=tool_calls or [],
            response_metadata={},
        )
        return response
    return _make


@pytest.fixture
def rag_service():
    from rag.rag_service import RagSummarizeService
    return RagSummarizeService(retriever_mode="vector")
