"""
=============================================================================
 two_retriever_mode_test.py —— 问答质量对比测试（检索 + LLM 生成回答）

 用途：对比 纯向量模式(vector) 和 混合模式(hybrid) 的完整客服问答质量。
      先用检索器取回文档，再交给 LLM 总结生成回答，评估最终回答质量。
      （对应 test_retrieval.py 则是仅对比检索出的原始文档差异，不含 LLM 生成）
=============================================================================
"""
import path_setup

from rag.rag_service import RagSummarizeService
import time

test_questions = [
    "小户型适合什么扫地机器人？",
    "E05 报错怎么解决？",
    "拖地功能怎么用？",
    "2000 元预算有什么推荐？",
    "电池续航一般多久？",
]


def test_mode(mode_name: str, retriever_mode: str):
    print(f"\n{'='*60}")
    print(f"{mode_name}")
    print(f"{'='*60}")

    rag = RagSummarizeService(retriever_mode=retriever_mode)

    for q in test_questions:
        print(f"\n>>> 问题: {q}")
        start = time.time()
        try:
            answer = rag.rag_summarize(q)
            elapsed = time.time() - start
            print(f"[耗时 {elapsed:.1f}s]")
            print(answer)
        except Exception as e:
            print(f"[错误] {e}")


if __name__ == "__main__":
    test_mode("纯向量检索模式 (retriever_mode='vector')", "vector")
    print("\n\n")
    test_mode("混合检索模式 (retriever_mode='hybrid')", "hybrid")
