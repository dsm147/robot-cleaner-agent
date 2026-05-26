"""工具函数测试"""
from utils.path_tool import get_project_root, get_abs_path
from utils.config_handler import rag_conf, chroma_conf


class TestPathTool:
    def test_get_project_root(self):
        root = get_project_root()
        assert "Agent(RAG)项目" in root

    def test_get_abs_path(self):
        path = get_abs_path("config/rag.yml")
        assert path.endswith("config/rag.yml")
        assert "Agent(RAG)项目" in path


class TestConfigHandler:
    def test_rag_conf_has_chat_model(self):
        assert "chat_model_name" in rag_conf

    def test_chroma_conf_has_k(self):
        assert chroma_conf["k"] > 0
