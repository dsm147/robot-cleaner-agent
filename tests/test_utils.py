"""工具函数测试"""
import os
from utils.path_tool import get_project_root, get_abs_path
from utils.config_handler import rag_conf, chroma_conf


class TestPathTool:
    def test_get_project_root(self):
        root = get_project_root()
        assert os.path.isdir(root)
        # 项目根目录下应存在关键目录
        assert os.path.isdir(os.path.join(root, "app"))
        assert os.path.isdir(os.path.join(root, "config"))

    def test_get_abs_path(self):
        path = get_abs_path("config/rag.yml")
        assert path.endswith("config/rag.yml")
        assert os.path.isfile(path)


class TestConfigHandler:
    def test_rag_conf_has_chat_model(self):
        assert "chat_model_name" in rag_conf

    def test_chroma_conf_has_k(self):
        assert chroma_conf["k"] > 0
