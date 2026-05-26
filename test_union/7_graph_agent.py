"""
LangGraph 版 Agent：用 StateGraph 重构 ReAct Agent
"""
import path_setup
import json
from typing import TypedDict, Annotated, Literal

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from model.factory import chat_model
from agent.manual_agent import TOOL_DEFINITIONS, TOOL_REGISTRY
from utils.prompt_loader import load_system_prompts


# ======== 定义 State ========

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_count: int
    max_iterations: int


# ======== 定义节点 ========

def call_model(state: AgentState) -> dict:
    messages = state["messages"]
    count = state.get("tool_call_count", 0)
    max_iter = state.get("max_iterations", 5)

    print(f"[call_model] 当前消息数: {len(messages)}，已调用工具: {count} 次")

    if count >= max_iter:
        print("[call_model] 超过最大迭代次数")
        return {
            "messages": [AIMessage(content="抱歉，我无法在有限的步骤内回答这个问题。")],
        }

    response = chat_model.invoke(messages, tools=TOOL_DEFINITIONS)

    ai_msg = AIMessage(
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
    )

    return {"messages": [ai_msg]}


def execute_tool(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_calls = []

    if hasattr(last_message, "additional_kwargs") and "tool_calls" in last_message.additional_kwargs:
        tool_calls = last_message.additional_kwargs["tool_calls"]

    if not tool_calls:
        return {"messages": [], "tool_call_count": state["tool_call_count"]}

    tool_messages = []

    for tc in tool_calls:
        tool_name = tc["function"]["name"]
        tool_id = tc["id"]

        try:
            tool_args = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            tool_args = {}

        print(f"[execute_tool] 调用: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")

        if tool_name in TOOL_REGISTRY:
            try:
                result = TOOL_REGISTRY[tool_name](**tool_args)
            except Exception as e:
                result = f"工具调用失败: {e}"
        else:
            result = f"错误: 未知工具 {tool_name}"

        tool_messages.append(ToolMessage(
            content=str(result)[:2000],
            tool_call_id=tool_id,
        ))

    return {
        "messages": tool_messages,
        "tool_call_count": state["tool_call_count"] + len(tool_calls),
    }


# ======== 路由条件 ========

def should_continue(state: AgentState) -> Literal["execute_tool", "__end__"]:
    last_message = state["messages"][-1]

    has_tools = False
    if hasattr(last_message, "additional_kwargs") and "tool_calls" in last_message.additional_kwargs:
        has_tools = bool(last_message.additional_kwargs["tool_calls"])

    if state["tool_call_count"] >= state["max_iterations"]:
        print("[should_continue] 超过最大迭代，结束")
        return "__end__"

    if has_tools:
        return "execute_tool"
    return "__end__"


# ======== 构建图 ========

def build_agent_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("call_model", call_model)
    graph.add_node("execute_tool", execute_tool)

    graph.set_entry_point("call_model")

    graph.add_conditional_edges(
        "call_model",
        should_continue,
    )

    graph.add_edge("execute_tool", "call_model")

    return graph


# ======== Agent 封装 ========

class GraphReActAgent:
    """LangGraph 版的 ReAct Agent"""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.system_prompt = load_system_prompts()
        self.graph = build_agent_graph()
        self.app = self.graph.compile()

    def execute(self, user_input: str) -> str:
        initial_state = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                HumanMessage(content=user_input),
            ],
            "tool_call_count": 0,
            "max_iterations": self.max_iterations,
        }

        final_state = None
        for state in self.app.stream(initial_state):
            final_state = state

        if final_state:
            last_node_output = final_state[list(final_state.keys())[-1]]
            messages = last_node_output.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    return last_msg.content or ""

        return "Agent 执行完毕"

    def execute_stream(self, user_input: str):
        yield self.execute(user_input)


if __name__ == "__main__":
    agent = GraphReActAgent()

    print("="*60)
    print("LangGraph 版 Agent 测试")
    print("="*60)

    test_inputs = [
        "小户型适合什么机器人？",
    ]

    for inp in test_inputs:
        print(f"\n>>> 用户: {inp}")
        print(">>> Agent: ", end="", flush=True)
        result = agent.execute(inp)
        print(result[:200])
        print()
