"""
对比测试四种检索模式：vector / hybrid / rewrite / hyde
"""
import path_setup

from rag.rag_service import RagSummarizeService


def test_modes():
    test_questions = [
        "这东西能自己拖地吗？",
        "噪音大不大？",
        "养猫适合什么机器人？",
        "电池能用多久？",
    ]

    modes = ["vector", "hybrid", "rewrite", "hyde"]

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"检索模式: {mode}")
        print(f"{'=' * 60}")
        try:
            rag = RagSummarizeService(retriever_mode=mode)
            for q in test_questions:
                print(f"\n问题: {q}")
                answer = rag.rag_summarize(q)
                print(f"回答: {answer[:150]}...")
        except Exception as e:
            print(f"模式 {mode} 出错: {e}")


if __name__ == "__main__":
    test_modes()
