"""
对比测试：单体 Agent vs Multi-Agent
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.react_agent import ReactAgent
from agent.orchestrator import MultiAgentSystem


def load_test_dataset() -> list[dict]:
    dataset_path = Path(__file__).parent / "test_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_agent(name: str, agent, test_cases: list[dict]) -> dict:
    results = []

    for case in test_cases:
        question = case["question"]

        print(f"  [{name}] {question[:40]}...")

        start = time.time()
        response = agent.execute(question)
        elapsed = time.time() - start

        results.append({
            "question": question,
            "response_length": len(response),
            "elapsed": round(elapsed, 2),
            "category": case.get("category", ""),
        })

    total = len(results)
    avg_time = sum(r["elapsed"] for r in results) / total if total > 0 else 0
    avg_length = sum(r["response_length"] for r in results) / total if total > 0 else 0

    return {
        "agent_name": name,
        "total_cases": total,
        "avg_response_time": round(avg_time, 2),
        "avg_response_length": round(avg_length, 1),
        "details": results,
    }


if __name__ == "__main__":
    dataset = load_test_dataset()

    print("初始化单体 Agent...")
    mono_agent = ReactAgent()

    print("初始化 Multi-Agent...")
    multi_agent = MultiAgentSystem()

    print("\n" + "="*60)
    print("评估单体 Agent")
    print("="*60)
    mono_results = evaluate_agent("单体Agent", mono_agent, dataset)

    print("\n" + "="*60)
    print("评估 Multi-Agent")
    print("="*60)
    multi_results = evaluate_agent("Multi-Agent", multi_agent, dataset)

    print("\n\n" + "="*60)
    print("对比结果")
    print("="*60)
    print(f"{'指标':<25} {'单体Agent':<15} {'Multi-Agent':<15}")
    print("-"*55)
    print(f"{'测试用例数':<25} {mono_results['total_cases']:<15} {multi_results['total_cases']:<15}")
    print(f"{'平均响应时间(s)':<25} {mono_results['avg_response_time']:<15} {multi_results['avg_response_time']:<15}")
    print(f"{'平均响应长度(字)':<25} {mono_results['avg_response_length']:<15} {multi_results['avg_response_length']:<15}")

    output = {
        "mono_agent": mono_results,
        "multi_agent": multi_results,
        "comparison": {
            "time_diff": round(multi_results["avg_response_time"] - mono_results["avg_response_time"], 2),
            "length_diff": round(multi_results["avg_response_length"] - mono_results["avg_response_length"], 1),
        }
    }

    output_path = Path(__file__).parent / "results" / "mono_vs_multi_comparison.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n对比结果已保存到: {output_path}")
