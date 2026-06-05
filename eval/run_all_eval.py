"""
批量运行所有评估，生成对比报告
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_rag_eval():
    from eval.eval_rag import RAGEvaluator, load_test_dataset

    dataset = load_test_dataset()
    rag_cases = [c for c in dataset if c["category"] != "报告生成"]

    configs = [
        {"mode": "vector", "reranker": False, "name": "基础版"},
        {"mode": "hybrid", "reranker": False, "name": "混合检索"},
        {"mode": "rewrite", "reranker": False, "name": "Query Rewrite"},
        {"mode": "hybrid", "reranker": True, "name": "混合+Reranker"},
    ]

    all_results = []
    for cfg in configs:
        print(f"\n>>> 运行 RAG 评估: {cfg['name']}")
        evaluator = RAGEvaluator(retriever_mode=cfg["mode"], use_reranker=cfg["reranker"])
        result = evaluator.evaluate(rag_cases)
        result["name"] = cfg["name"]
        all_results.append(result)

    print("\n\n========== RAG 评估汇总 ==========")
    for r in all_results:
        print(f"{r['name']}: {r['mode']}")

    return all_results


def run_agent_eval():
    from eval.eval_agent import AgentEvaluator, load_test_dataset, save_results

    dataset = load_test_dataset()
    evaluator = AgentEvaluator(agent_name="LangChain Agent")
    results = evaluator.evaluate(dataset)
    save_results(results, f"agent_eval_{int(time.time())}.json")

    print(f"\nAgent 评估结果:")
    print(f"  工具调用准确率: {results['tool_call_accuracy']:.2%}")
    print(f"  平均响应时间: {results['avg_response_time']}s")

    return results


if __name__ == "__main__":
    print("="*60)
    print("开始全面评估")
    print("="*60)

    rag_results = run_rag_eval()
    agent_results = run_agent_eval()

    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)
