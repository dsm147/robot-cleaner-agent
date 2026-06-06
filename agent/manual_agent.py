"""
手写 ReAct Agent：不使用 LangChain Agent 框架，从零实现
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.factory import chat_model
from agent.tools.agent_tools import (
    rag_summarize, get_weather, get_user_location,
    get_user_id, get_current_month, fetch_external_data,
    fill_context_for_report,
)
from utils.prompt_loader import load_system_prompts
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


# ======== 工具注册 ========

def rag_summarize_func(query: str) -> str:
    return rag_summarize.invoke({"query": query})


def get_weather_func(city: str) -> str:
    return get_weather.invoke({"city": city})


def get_user_location_func() -> str:
    return get_user_location.invoke({})


def get_user_id_func() -> str:
    return get_user_id.invoke({})


def get_current_month_func() -> str:
    return get_current_month.invoke({})


def fetch_external_data_func(user_id: str, month: str) -> str:
    return fetch_external_data.invoke({"user_id": user_id, "month": month})


def fill_context_for_report_func() -> str:
    return fill_context_for_report.invoke({})

# 工具定义列表
# 注意：每个工具都必须在 TOOL_REGISTRY 中有对应的实现函数
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "rag_summarize",
            "description": "从向量存储中检索参考资料",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_location",
            "description": "获取用户所在城市的名称",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_id",
            "description": "获取用户的ID",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_month",
            "description": "获取当前月份",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_external_data",
            "description": "获取指定用户在指定月份的使用记录",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "month": {"type": "string"},
                },
                "required": ["user_id", "month"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fill_context_for_report",
            "description": "为报告生成注入上下文（报告场景必须调用）",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

# 工具注册表：工具名称 -> 实现函数
TOOL_REGISTRY = {
    "rag_summarize": rag_summarize_func,
    "get_weather": get_weather_func,
    "get_user_location": get_user_location_func,
    "get_user_id": get_user_id_func,
    "get_current_month": get_current_month_func,
    "fetch_external_data": fetch_external_data_func,
    "fill_context_for_report": fill_context_for_report_func,
}

# 工具定义映射：工具名称 -> 工具定义（方便根据名称动态构建工具列表）
TOOL_DEFINITIONS_MAP = {
    tool_def["function"]["name"]: tool_def
    for tool_def in TOOL_DEFINITIONS
}


# ======== 工具执行辅助函数 ========

def execute_tool_with_retry(tool_name: str, tool_args: dict, max_retries: int = 2) -> str:
    """执行工具（带重试）"""
    last_error = ""

    for attempt in range(max_retries + 1):
        try:
            if tool_name not in TOOL_REGISTRY:
                return f"错误: 未知工具 {tool_name}"

            if attempt > 0:
                print(f"[重试] {tool_name} 第 {attempt + 1} 次尝试")

            result = TOOL_REGISTRY[tool_name](**tool_args)
            return str(result)

        except Exception as e:
            last_error = str(e)
            print(f"[工具错误] {tool_name} 第 {attempt + 1} 次失败: {e}")

    return f"工具 {tool_name} 调用失败（已重试 {max_retries} 次）: {last_error}"


# ======== Agent 核心循环 ========

class ManualReactAgent:
    """
    手写 ReAct Agent
    完全不用 LangChain Agent 框架，只用底层的 ChatTongyi SDK
    """

    def __init__(self, max_iterations: int = 5, max_history: int = 4, system_prompt_path: str = None):
        self.max_iterations = max_iterations
        self.max_history = max_history
        if system_prompt_path:
            self.system_prompt = self._load_prompt(system_prompt_path)
        else:
            self.system_prompt = load_system_prompts()
        self.history: list[dict] = []
        self.system_prompt_path = system_prompt_path

    def _load_prompt(self, path: str) -> str:
        full_path = get_abs_path(path)
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

    def _build_tool_definitions(self) -> list[dict]:
        """根据 Agent 类型动态构建工具定义"""
        if self.system_prompt_path and "report" in self.system_prompt_path:
            return [
                TOOL_DEFINITIONS_MAP[name]
                for name in ["get_user_id", "get_current_month",
                             "fill_context_for_report", "fetch_external_data"]
                if name in TOOL_DEFINITIONS_MAP
            ]
        elif self.system_prompt_path and "customer_service" in self.system_prompt_path:
            return [
                TOOL_DEFINITIONS_MAP[name]
                for name in ["rag_summarize", "get_weather", "get_user_location"]
                if name in TOOL_DEFINITIONS_MAP
            ]
        return TOOL_DEFINITIONS

    def _build_messages(self, user_input: str) -> list[dict]:
        """构建包含历史的消息列表"""
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in self.history[-self.max_history:]:
            messages.append(msg)
        messages.append({"role": "user", "content": user_input})
        return messages

    def _update_history(self, messages: list[dict]):
        """从完整消息列表中提取并保存对话历史"""
        conversation = messages[1:]
        user_msg = None
        assistant_msg = None

        for msg in conversation:
            if msg["role"] == "user":
                user_msg = msg
            elif msg["role"] == "assistant" and not msg.get("tool_calls"):
                assistant_msg = msg

        if user_msg:
            self.history.append(user_msg)
        if assistant_msg:
            self.history.append(assistant_msg)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def _detect_loop(self, tool_call_history: list[dict]) -> bool:
        """检测是否陷入死循环"""
        if len(tool_call_history) < 4:
            return False

        recent = tool_call_history[-4:]
        names = [call["name"] for call in recent]
        if len(set(names)) == 1:
            return True

        return False

    def estimate_tokens(self, messages: list[dict]) -> int:
        total = 0
        for msg in messages:
            total += len(str(msg.get("content", ""))) // 2
        return total

    def execute(self, user_input: str) -> str:
        messages = self._build_messages(user_input)
        tool_call_history = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n[迭代 {iteration}/{self.max_iterations}]")
            print(f"[LLM 调用] messages 共 {len(messages)} 条, 约 {self.estimate_tokens(messages)} tokens")

            response = chat_model.invoke(messages, tools=self._build_tool_definitions())

            assistant_msg = {
                "role": "assistant",
                "content": response.content,
            }

            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls_data = []
                for tc in response.tool_calls:
                    tool_call = {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"], ensure_ascii=False),
                        },
                    }
                    tool_calls_data.append(tool_call)
                    tool_call_history.append({"name": tc["name"], "args": tc["args"]})

                # 死循环检测
                if self._detect_loop(tool_call_history):
                    print("[检测] Agent 陷入死循环，中断")
                    return "我似乎陷入了循环，请换一种方式提问。"

                assistant_msg["tool_calls"] = tool_calls_data
                messages.append(assistant_msg)

                for tc in response.tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["args"]

                    if tool_name in TOOL_REGISTRY:
                        print(f"[工具调用] {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")
                        result = execute_tool_with_retry(tool_name, tool_args)
                        print(f"[工具结果] {str(result)[:100]}...")

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": str(result)[:2000],
                        })
                    else:
                        print(f"[工具错误] 未知工具: {tool_name}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": f"错误: 未知工具 {tool_name}",
                        })
            else:
                final_answer = response.content or ""
                print(f"[最终回答] {final_answer[:100]}...")
                self._update_history(messages)
                return final_answer

        return "抱歉，我无法在有限的步骤内回答这个问题。"

    def execute_stream(self, user_input: str):
        result = self.execute(user_input)
        yield result


if __name__ == "__main__":
    agent = ManualReactAgent()

    print("="*60)
    print("手写 ReAct Agent 测试")
    print("="*60)

    test_inputs = [
        "小户型适合什么机器人？",
    ]

    for inp in test_inputs:
        print(f"\n>>> 用户: {inp}")
        print(">>> Agent: ", end="", flush=True)
        for chunk in agent.execute_stream(inp):
            print(chunk, end="", flush=True)
        print()
