"""
yaml
k: v
"""
import yaml
from utils.path_tool import get_abs_path

# 加载配置文件的工具函数，分别加载不同的配置文件，返回一个字典
def load_rag_config(config_path: str=get_abs_path("config/rag.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_chroma_config(config_path: str=get_abs_path("config/chroma.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str=get_abs_path("config/agent.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

# 加载配置文件，放在这里是为了全局只加载一次，后续需要用到配置的地方直接导入这个模块就能拿到配置了
rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()

# 测试代码，直接运行这个文件就能看到效果
if __name__ == '__main__':
    print(rag_conf["chat_model_name"])
