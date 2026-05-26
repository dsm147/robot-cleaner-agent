"""
RAG 量化评估：Hit Rate、MRR、Faithfulness 等指标
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.rag_service import RagSummarizeService
from rag.vector_store import VectorStoreService


def evaluate_retrieval():
    """评估检索质量：Hit Rate、MRR"""
    dataset_path = Path(__file__).parent / "test_dataset.json"
    import json
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    store = VectorStoreService()

    configs = [
        ("纯向量", store.get_retriever(search_k=5)),
        ("混合检索", store.get_hybrid_retriever(retriever_k=5)),
    ]

    for name, retriever in configs:
        print(f"\n{'='*60}")
        print(f"检索模式: {name}")
        print(f"{'='*60}")

        total = 0
        hit = 0
        reciprocal_ranks = []

        for case in test_cases:
            question = case["question"]
            expected = case.get("expected_answer", "")

            if not expected:
                continue

            total += 1
            docs = retriever.invoke(question)

            found = False
            for rank, doc in enumerate(docs, 1):
                if expected in doc.page_content:
                    hit += 1
                    found = True
                    reciprocal_ranks.append(1.0 / rank)
                    break

            if not found:
                reciprocal_ranks.append(0.0)

            status = "✓" if found else "✗"
            print(f"  [{status}] {question[:25]:<25} -> {'找到' if found else '未找到'}")

        hit_rate = hit / total if total > 0 else 0
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0

        print(f"\n结果汇总 ({name}):")
        print(f"  Hit Rate@{5}: {hit_rate:.2%} ({hit}/{total})")
        print(f"  MRR:          {mrr:.4f}")


def evaluate_faithfulness():
    """评估生成质量：Faithfulness、Answer Relevancy（需要 RAGAS）"""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
    except ImportError:
        print("\n[跳过] 需要安装 ragas 和 datasets: pip install ragas datasets")
        return

    import json
    dataset_path = Path(__file__).parent / "test_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    questions = [c["question"] for c in test_cases[:5]]

    rag = RagSummarizeService(retriever_mode="hybrid")

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
    }

    for q in questions:
        print(f"评估: {q[:40]}...")
        docs = rag.retriever_docs(q)
        answer = rag.rag_summarize(q)

        data["question"].append(q)
        data["answer"].append(answer)
        data["contexts"].append([d.page_content for d in docs])

    dataset = Dataset.from_dict(data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
    )

    print(f"\n生成质量评估:")
    print(f"  Faithfulness:     {result['faithfulness']:.4f}")
    print(f"  Answer Relevancy: {result['answer_relevancy']:.4f}")

    return result


if __name__ == "__main__":
    print("="*60)
    print("RAG 量化评估")
    print("="*60)

    evaluate_retrieval()
    print("\n")
    evaluate_faithfulness()
