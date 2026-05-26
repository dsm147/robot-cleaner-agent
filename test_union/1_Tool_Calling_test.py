"""
Tool Calling 测试
"""
import path_setup

import json
from agent.react_agent import ReactAgent
from model.factory import chat_model


def debug_invoke(messages, tools):
    """打印完整的 LLM 请求体"""
    print("\n" + "="*60)
    print("[DEBUG] 发送给 LLM 的 messages:")
    print("="*60)
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        # 只打印前 200 个字符，避免刷屏
        print(f"  [{role}]: {str(content)[:200]}")
        if "tool_calls" in msg:
            print(f"  [tool_calls]: {json.dumps(msg['tool_calls'], ensure_ascii=False, indent=2)[:300]}")
    print("="*60)


# 测试：直接构造一次调用，观察完整的请求和响应
if __name__ == "__main__":
    # 观察 Tool Calling
    print("\n>>> 测试 1：直接调用 LLM，观察请求/响应格式\n")

    messages = [
        {"role": "system", "content": "你是一个只能调用工具的助手，不能自己回答。"},
        {"role": "user", "content": "深圳今天天气怎么样？"}
    ]

    # 定义一个工具
    tools = [
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
                            "description": "城市名"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    print("请求 messages:", json.dumps(messages, ensure_ascii=False, indent=2))
    print("\n请求 tools:", json.dumps(tools, ensure_ascii=False, indent=2))

    # 直接调用底层模型
    response = chat_model.invoke(messages, tools=tools) # type: ignore

    print("\n响应:")
    print(json.dumps({
        "content": response.content,
        "tool_calls": response.tool_calls,
        "response_metadata": response.response_metadata,
    }, ensure_ascii=False, indent=2))

    if response.tool_calls:
        print("\n>>> 检测到 Tool Call!")
        for tc in response.tool_calls:
            print(f"  工具名: {tc['name']}")
            print(f"  参数: {json.dumps(tc['args'], ensure_ascii=False, indent=2)}")