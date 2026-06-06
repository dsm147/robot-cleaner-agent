import sys
import os
from typing import cast, Any
from collections.abc import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import create_agent
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=cast(BaseChatModel, chat_model),
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=cast(
                Sequence[AgentMiddleware[AgentState[Any], Any, Any]],
                [monitor_tool, log_before_model, report_prompt_switch],
            ),
            context_schema=dict,
        )
        self.last_tool_calls: list[str] = []

    def execute_stream(self, query: str):
        self.last_tool_calls = []
        input_dict = {
            "messages": [HumanMessage(content=query)],
        }

        last_content = ""# 记录上一次 yield 的内容，避免重复 yield 同一内容（可能因为 stream_mode="values" 在 agent 结束时重复 yield 同一状态）

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记（report_prompt_switch这个middleware会根据这个标记来切换提示词），我们这里放了一个report字段，表示是否是生成报告的流程，初始为False，在生成报告的提示词里会把这个字段设为True，这样在agent执行过程中我们就能知道当前是不是在生成报告了，这个信息在工具函数fill_context_for_report里会用到，如果正在生成报告就会把agent的状态信息放到上下文里，这样工具函数就能获取到agent的状态信息来填充报告内容）
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]

            # 判断是不是工具调用信息
            # 检测 AI 是否调用了工具，把工具名记录下来(先判断是不是AImessage，因为HumanMessage没有additional_kwargs属性)
            if hasattr(latest_message, "additional_kwargs") and "tool_calls" in latest_message.additional_kwargs:
                # if如果为真就代表调用了工具，工具调用的信息在additional_kwargs的tool_calls字段里，是一个列表，列表里每个元素是一个工具调用的信息字典，字典里有个function字段，里面有个name字段就是工具名
                for tc in latest_message.additional_kwargs["tool_calls"]:
                    name = tc.get("function", {}).get("name", "")
                    if name and name not in self.last_tool_calls:
                        self.last_tool_calls.append(name)

            # 判断是不是最终回复信息
            # 只 yield AI 的回复，过滤 HumanMessage（START 节点的初始状态）和 ToolMessage
            if isinstance(latest_message, AIMessage) and latest_message.content:
                content = latest_message.content.strip()
                # stream_mode="values" 在 agent 结束时可能重复 yield 同一状态
                if content and content != last_content:
                    last_content = content
                    yield content + "\n"

# 测试代码，直接运行这个文件就能看到效果
if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
