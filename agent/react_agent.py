import sys
import os
from typing import cast, Any
from collections.abc import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import create_agent
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
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

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if hasattr(latest_message, "additional_kwargs") and "tool_calls" in latest_message.additional_kwargs:
                for tc in latest_message.additional_kwargs["tool_calls"]:
                    name = tc.get("function", {}).get("name", "")
                    if name and name not in self.last_tool_calls:
                        self.last_tool_calls.append(name)
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
