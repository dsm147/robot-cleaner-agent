"""
LangGraph 版 Multi-Agent
使用 StateGraph 实现 Orchestrator + Worker 调度
"""
import path_setup
import json
import re
from typing import TypedDict, Annotated, Literal

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from model.factory import chat_model
from utils.path_tool import get_abs_path


def load_prompt(path):
    with open(get_abs_path(path), "r", encoding="utf-8") as f:
        return f.read()


ORCHESTRATOR_PROMPT = load_prompt("prompts/orchestrator_prompt.txt")
CS_PROMPT = load_prompt("prompts/customer_service_prompt.txt")
REPORT_PROMPT = load_prompt("prompts/report_agent_prompt.txt")


# ======== State ========

class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str
    intent: str
    confidence: float
    current_agent: str


# ======== 节点 ========

def orchestrator_node(state: MultiAgentState) -> dict:
    user_input = state["user_input"]

    response = chat_model.invoke([
        SystemMessage(content=ORCHESTRATOR_PROMPT),
        HumanMessage(content=user_input),
    ])

    content = response.content.strip()

    try:
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(content)
    except (json.JSONDecodeError, AttributeError):
        result = {"intent": "customer_service", "confidence": 0.5}

    print(f"[Orchestrator] 意图: {result.get('intent')}, 置信度: {result.get('confidence', 0):.2f}")

    return {
        "intent": result.get("intent", "customer_service"),
        "confidence": result.get("confidence", 0.5),
        "current_agent": result.get("intent", "customer_service"),
    }


def customer_service_node(state: MultiAgentState) -> dict:
    from agent.manual_agent import TOOL_DEFINITIONS_MAP, TOOL_REGISTRY

    user_input = state["user_input"]
    cs_tool_defs = [
        TOOL_DEFINITIONS_MAP[name]
        for name in ["rag_summarize", "get_weather", "get_user_location"]
        if name in TOOL_DEFINITIONS_MAP
    ]

    messages = [
        SystemMessage(content=CS_PROMPT),
        HumanMessage(content=user_input),
    ]

    for _ in range(5):
        response = chat_model.invoke(messages, tools=cs_tool_defs)

        if hasattr(response, "tool_calls") and response.tool_calls:
            messages.append(AIMessage(
                content=response.content or "",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"], ensure_ascii=False),
                            },
                            "type": "function",
                        }
                        for tc in (response.tool_calls or [])
                    ]
                } if response.tool_calls else {},
            ))

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]

                if tool_name in TOOL_REGISTRY:
                    try:
                        result = TOOL_REGISTRY[tool_name](**tool_args)
                    except Exception as e:
                        result = f"工具调用失败: {e}"
                else:
                    result = f"未知工具: {tool_name}"

                messages.append(ToolMessage(content=str(result)[:2000], tool_call_id=tc["id"]))
        else:
            return {"messages": [AIMessage(content=response.content or "")]}

    return {"messages": [AIMessage(content="无法完成请求。")]}


def report_agent_node(state: MultiAgentState) -> dict:
    from agent.manual_agent import TOOL_DEFINITIONS_MAP, TOOL_REGISTRY

    user_input = state["user_input"]
    report_tool_defs = [
        TOOL_DEFINITIONS_MAP[name]
        for name in ["get_user_id", "get_current_month",
                      "fill_context_for_report", "fetch_external_data"]
        if name in TOOL_DEFINITIONS_MAP
    ]

    messages = [
        SystemMessage(content=REPORT_PROMPT),
        HumanMessage(content=user_input),
    ]

    for _ in range(6):
        response = chat_model.invoke(messages, tools=report_tool_defs)

        if hasattr(response, "tool_calls") and response.tool_calls:
            messages.append(AIMessage(
                content=response.content or "",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"], ensure_ascii=False),
                            },
                            "type": "function",
                        }
                        for tc in (response.tool_calls or [])
                    ]
                } if response.tool_calls else {},
            ))

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]

                if tool_name in TOOL_REGISTRY:
                    try:
                        result = TOOL_REGISTRY[tool_name](**tool_args)
                    except Exception as e:
                        result = f"工具调用失败: {e}"
                else:
                    result = f"未知工具: {tool_name}"

                messages.append(ToolMessage(content=str(result)[:2000], tool_call_id=tc["id"]))
        else:
            return {"messages": [AIMessage(content=response.content or "")]}

    return {"messages": [AIMessage(content="无法生成报告。")]}


# ======== 路由 ========

def route_worker(state: MultiAgentState) -> Literal["customer_service", "report", "__end__"]:
    intent = state.get("intent", "customer_service")
    if intent == "report":
        return "report"
    return "customer_service"


# ======== 构建图 ========

def build_multi_agent_graph() -> StateGraph:
    graph = StateGraph(MultiAgentState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("customer_service", customer_service_node)
    graph.add_node("report", report_agent_node)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        route_worker,
    )

    graph.add_edge("customer_service", END)
    graph.add_edge("report", END)

    return graph


class GraphMultiAgent:
    """LangGraph 版 Multi-Agent"""

    def __init__(self):
        self.graph = build_multi_agent_graph()
        self.app = self.graph.compile()

    def execute(self, user_input: str) -> str:
        initial_state = {
            "messages": [],
            "user_input": user_input,
            "intent": "",
            "confidence": 0.0,
            "current_agent": "",
        }

        final_state = None
        for state in self.app.stream(initial_state):
            final_state = state

        if final_state:
            last_node = list(final_state.keys())[-1]
            node_output = final_state[last_node]
            messages = node_output.get("messages", [])
            if messages:
                return messages[-1].content or ""

        return "处理完毕"

    def execute_stream(self, user_input: str):
        yield self.execute(user_input)


if __name__ == "__main__":
    agent = GraphMultiAgent()

    test_cases = [
        "小户型适合什么机器人？",
        "E05 报错什么意思？",
        "生成我的使用报告",
        "深圳今天天气怎么样？",
    ]

    for inp in test_cases:
        print(f"\n>>> 用户: {inp}")
        print(">>> 系统: ", end="", flush=True)
        result = agent.execute(inp)
        print(result[:200])
        print()
