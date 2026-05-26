"""RAG 模块测试"""


class TestRagService:
    """RagSummarizeService 基础功能测试"""

    def test_init(self, rag_service):
        assert rag_service is not None
        assert rag_service.retriever_mode == "vector"

    def test_retriever_mode_switch(self):
        from rag.rag_service import RagSummarizeService
        for mode in ["vector", "hybrid", "rewrite", "hyde"]:
            rag = RagSummarizeService(retriever_mode=mode)
            assert rag.retriever_mode == mode

    def test_retriever_docs_returns_list(self, rag_service):
        docs = rag_service.retriever_docs("测试查询")
        assert isinstance(docs, list)

    def test_summarize_returns_string(self, rag_service):
        result = rag_service.rag_summarize("小户型适合什么机器人？")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summarize_with_reranker(self):
        from rag.rag_service import RagSummarizeService
        rag = RagSummarizeService(retriever_mode="hybrid", use_reranker=True)
        result = rag.rag_summarize("E05 报错")
        assert isinstance(result, str)
