import os
from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf


def _get_api_key() -> str:
    """优先从环境变量读取 API key，否则从 YAML 配置文件读取"""
    key = os.environ.get("DASHSCOPE_API_KEY")
    if key:
        return key
    # 兼容旧版：配置文件中仍有 api_key 字段时使用
    return rag_conf.get("api_key", "")


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model=rag_conf["chat_model_name"], api_key=_get_api_key())


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"], dashscope_api_key=_get_api_key())


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
