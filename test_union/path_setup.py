"""
所有 test_union/ 下测试脚本共享的路径初始化。
导入此模块即可将项目根目录加入 Python 路径，避免 ModuleNotFoundError。
"""
import sys
import os

# 本文件在 test_union/ 下，向上退一级就是项目根目录
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
