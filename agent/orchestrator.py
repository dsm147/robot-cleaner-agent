"""
Multi-Agent 调度器（Orchestrator）
负责：意图分类 → 分派给对应 Worker Agent → 返回结果
"""
import json
import re
from model.factory import chat_model
from utils.path_tool import get_abs_path


def load_prompt(file_path: str) -> str:
    path = get_abs_path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


ORCHESTRATOR_PROMPT = load_prompt("prompts/orchestrator_prompt.txt")


class OrchestratorAgent:
    """调度 Agent：分析意图并分派任务"""

    def classify(self, user_input: str) -> dict:
        response = chat_model.invoke([
            {"role": "system", "content": ORCHESTRATOR_PROMPT},
            {"role": "user", "content": user_input},
        ])

        content = response.content.strip()

        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(content)

            return {
                "intent": result.get("intent", "customer_service"),
                "confidence": result.get("confidence", 0.5),
                "reason": result.get("reason", ""),
                "user_input": user_input,
            }
        except (json.JSONDecodeError, AttributeError):
            return {
                "intent": "customer_service",
                "confidence": 0.5,
                "reason": "意图解析失败，默认走客服",
                "user_input": user_input,
            }


class CustomerServiceAgent:
    """客服 Agent：处理产品咨询和故障排查"""

    def __init__(self):
        from agent.manual_agent import ManualReactAgent
        self.agent = ManualReactAgent(
            system_prompt_path="prompts/customer_service_prompt.txt",
            max_iterations=5,
        )

    def execute(self, user_input: str) -> str:
        return self.agent.execute(user_input)

    def execute_stream(self, user_input: str):
        yield self.execute(user_input)


class ReportAgent:
    """报告 Agent：生成使用报告"""

    def __init__(self):
        from agent.manual_agent import ManualReactAgent
        self.agent = ManualReactAgent(
            system_prompt_path="prompts/report_agent_prompt.txt",
            max_iterations=6,
        )

    def execute(self, user_input: str) -> str:
        return self.agent.execute(user_input)

    def execute_stream(self, user_input: str):
        yield self.execute(user_input)


class MultiAgentSystem:
    """
    Multi-Agent 系统入口
    Orchestrator + Worker 模式
    """

    def __init__(self):
        self.orchestrator = OrchestratorAgent()
        self.customer_service = CustomerServiceAgent()
        self.report = ReportAgent()

    def execute(self, user_input: str) -> str:
        print(f"[Orchestrator] 分析意图中...")
        dispatch = self.orchestrator.classify(user_input)
        print(f"[Orchestrator] 意图: {dispatch['intent']} (置信度: {dispatch['confidence']:.2f})")
        print(f"[Orchestrator] 理由: {dispatch['reason']}")

        if dispatch["intent"] == "report":
            print(f"[Orchestrator] → 分派给报告Agent")
            result = self.report.execute(user_input)
        else:
            print(f"[Orchestrator] → 分派给客服Agent")
            result = self.customer_service.execute(user_input)

        return result

    def execute_stream(self, user_input: str):
        yield self.execute(user_input)
