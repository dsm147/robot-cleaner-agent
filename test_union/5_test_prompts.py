"""
对比测试：原始提示词 vs Few-shot 增强提示词
"""
import path_setup
from rag.rag_service import RagSummarizeService

def test_prompt_versions():
    test_cases = [
        "X90 吸力多大？",
        "E05 报错什么意思？",
        "小户型适合什么机器人？",
        "拖地功能怎么用？",
        "电池续航多久？",
    ]

    print("="*60)
    print("当前 RAG 提示词版本（含 Few-shot）")
    print("="*60)
    rag = RagSummarizeService(retriever_mode="hybrid")
    for q in test_cases:
        print(f"\n问题: {q}")
        print(f"回答: {rag.rag_summarize(q)[:200]}")


if __name__ == "__main__":
    test_prompt_versions()
