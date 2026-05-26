"""
模型工厂：提供 Chat 和 Embedding 模型的懒加载单例
"""
import os
from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf


def _get_api_key() -> str:
    """从环境变量读取 DASHSCOPE_API_KEY"""
    key = os.environ.get("DASHSCOPE_API_KEY")
    if key:
        return key
    raise ValueError("DASHSCOPE_API_KEY 未设置！请通过环境变量配置：export DASHSCOPE_API_KEY=your_key")


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model=rag_conf["chat_model_name"], api_key=_get_api_key())


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(
            model=rag_conf["embedding_model_name"],
            dashscope_api_key=_get_api_key(),
        )


# ---- 懒加载实现 ----
# 使用模块级 __getattr__，首次访问 chat_model / embed_model 时才初始化
# 同时支持 unittest.mock.patch 正常 Mock

_chat_model = None
_embed_model = None


def get_chat_model():
    global _chat_model
    if _chat_model is None:
        _chat_model = ChatModelFactory().generator()
    return _chat_model


def get_embedding_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = EmbeddingsFactory().generator()
    return _embed_model


def __getattr__(name):
    if name == "chat_model":
        return get_chat_model()
    elif name == "embed_model":
        return get_embedding_model()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == '__main__':
    print(f"Chat Model: {get_chat_model()}")
    print(f"Embed Model: {get_embedding_model()}")
