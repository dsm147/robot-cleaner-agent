"""
Agent 评估脚本：评估任务完成率、工具调用准确率等
"""
import json
import sys
import time
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.react_agent import ReactAgent


class AgentEvaluator:
    """Agent 行为评估器"""

    def __init__(self, agent_name: str = "LangChain Agent"):
        self.agent = ReactAgent()
        self.agent_name = agent_name

    def evaluate(self, test_cases: list[dict]) -> dict:
        results = []

        for case in test_cases:
            question = case["question"]
            expected_tools = case.get("expected_tools", [])
            expected_answer = case.get("expected_answer", "")

            print(f"评估: {question[:50]}...")

            start = time.time()

            response = ""
            for chunk in self.agent.execute_stream(question):
                response += chunk

            elapsed = time.time() - start

            actual_tools = []

            results.append({
                "question": question,
                "response": response,
                "response_length": len(response),
                "actual_tools": actual_tools,
                "expected_tools": expected_tools,
                "expected_answer": expected_answer,
                "elapsed": elapsed,
            })

        total = len(results)
        avg_time = statistics.mean([r["elapsed"] for r in results]) if results else 0
        avg_length = statistics.mean([r["response_length"] for r in results]) if results else 0

        tool_correct = sum(
            1 for r in results
            if set(r["actual_tools"]) == set(r["expected_tools"])
        )
        tool_accuracy = tool_correct / total if total > 0 else 0

        keyword_hits = sum(
            1 for r in results
            if r["expected_answer"] and r["expected_answer"] in r["response"]
        )
        keyword_total = sum(1 for r in results if r["expected_answer"])
        keyword_accuracy = keyword_hits / keyword_total if keyword_total > 0 else 0

        return {
            "agent_name": self.agent_name,
            "total_cases": total,
            "avg_response_time": round(avg_time, 2),
            "avg_response_length": round(avg_length, 1),
            "tool_call_accuracy": round(tool_accuracy, 4),
            "keyword_accuracy": round(keyword_accuracy, 4),
            "details": results,
        }


def load_test_dataset() -> list[dict]:
    dataset_path = Path(__file__).parent / "test_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results: dict, filename: str):
    results_path = Path(__file__).parent / "results"
    results_path.mkdir(exist_ok=True)
    filepath = results_path / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到: {filepath}")


if __name__ == "__main__":
    dataset = load_test_dataset()

    evaluator = AgentEvaluator()
    results = evaluator.evaluate(dataset)

    print(f"\n{'='*60}")
    print(f"Agent 评估结果: {results['agent_name']}")
    print(f"{'='*60}")
    print(f"测试用例数: {results['total_cases']}")
    print(f"平均响应时间: {results['avg_response_time']}s")
    print(f"平均响应长度: {results['avg_response_length']} 字")
    print(f"工具调用准确率: {results['tool_call_accuracy']:.2%}")
    print(f"关键词命中率: {results['keyword_accuracy']:.2%}")

    save_results(results, f"agent_eval_{int(time.time())}.json")
