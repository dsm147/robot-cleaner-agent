"""
=============================================================================
 test_retrieval.py —— 检索结果对比测试（仅检索，不调用LLM生成回答）

 用途：对比 纯向量检索 和 混合检索(BM25+向量) 返回的原始文档差异。
      直接展示检索到的文档内容和来源，不经过 LLM 总结生成。
      （对应 test_compare_mode.py 则是包含 LLM 生成回答的完整问答质量对比）
=============================================================================
"""
import path_setup

from rag.vector_store import VectorStoreService
from utils.logger_handler import logger


def test_retrieval():
    store = VectorStoreService()

    # 确保向量库已加载文档，否则纯向量检索会返回空
    store.load_document()

    test_queries = [
        "小户型适合什么机器人",
        "E05 报错怎么解决",
        "拖地功能怎么用",
        "2000 元预算推荐",
        "电池续航多久",
    ]

    # 纯向量检索
    print("=" * 60)
    print("纯向量检索结果")
    print("=" * 60)
    vector_retriever = store.get_retriever()
    for query in test_queries:
        results = vector_retriever.invoke(query)
        print(f"\n查询: {query}")
        for i, doc in enumerate(results):
            print(f"  [{i + 1}] {doc.page_content[:80]}...")
            print(f"       来源: {doc.metadata}")

    # 混合检索
    print("\n" + "=" * 60)
    print("混合检索结果")
    print("=" * 60)
    hybrid_retriever = store.get_hybrid_retriever()
    for query in test_queries:
        results = hybrid_retriever.invoke(query)
        print(f"\n查询: {query}")
        for i, doc in enumerate(results):
            print(f"  [{i + 1}] {doc.page_content[:80]}...")
            print(f"       来源: {doc.metadata}")


if __name__ == "__main__":
    test_retrieval()
