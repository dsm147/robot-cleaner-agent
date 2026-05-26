"""
RAG 评估脚本：评估检索质量和生成质量
"""
import json
import sys
import time
from pathlib import Path

# 确保可以 import 项目模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.rag_service import RagSummarizeService


class RAGEvaluator:
    """RAG 质量评估器"""

    def __init__(self, retriever_mode: str, use_reranker: bool):
        self.rag = RagSummarizeService(
            retriever_mode=retriever_mode,
            use_reranker=use_reranker,
        )
        self.mode_name = f"{retriever_mode}_reranker={use_reranker}"

    def evaluate(self, test_cases: list[dict]) -> dict:
        questions = []
        answers = []
        contexts = []

        for case in test_cases:
            question = case["question"]
            print(f"评估: {question[:50]}...")

            start = time.time()
            answer = self.rag.rag_summarize(question)
            elapsed = time.time() - start

            docs = self.rag.retriever_docs(question)
            doc_texts = [doc.page_content for doc in docs]

            questions.append(question)
            answers.append(answer)
            contexts.append(doc_texts)

            print(f"  耗时: {elapsed:.2f}s, 回答长度: {len(answer)} 字")

        return {
            "mode": self.mode_name,
            "total_cases": len(test_cases),
            "avg_response_time": None,
        }


def load_test_dataset() -> list[dict]:
    dataset_path = Path(__file__).parent / "test_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    dataset = load_test_dataset()

    rag_cases = [c for c in dataset if c["category"] != "报告生成"]

    configs = [
        {"mode": "vector", "reranker": False},
        {"mode": "hybrid", "reranker": False},
        {"mode": "rewrite", "reranker": False},
        {"mode": "hybrid", "reranker": True},
    ]

    results = []
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"评估配置: {cfg}")
        print(f"{'='*60}")
        evaluator = RAGEvaluator(
            retriever_mode=cfg["mode"],
            use_reranker=cfg["reranker"],
        )
        result = evaluator.evaluate(rag_cases)
        results.append(result)
        print(f"\n结果: {result}")

    print("\n\n" + "="*60)
    print("汇总对比")
    print("="*60)
    for r in results:
        print(f"{r['mode']}")
