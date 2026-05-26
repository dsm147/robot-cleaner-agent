"""
Reranker：使用 Cross-encoder 对检索结果进行重排序
"""
from langchain_core.documents import Document


class CrossEncoderReranker:
    """Cross-encoder 重排序器"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self.model = None

    def _lazy_load_model(self):
        if self.model is None:
            from sentence_transformers import CrossEncoder
            print(f"[Reranker] 加载模型: {self.model_name}")
            self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 3,
    ) -> list[Document]:
        self._lazy_load_model()

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        top_docs = [doc for _, doc in scored_docs[:top_k]]

        print(f"\n[Reranker] 重排结果:")
        print(f"  输入 {len(documents)} 条，输出 {len(top_docs)} 条")
        for i, (score, doc) in enumerate(scored_docs[:top_k]):
            print(f"  [{i+1}] score={score:.4f} | {doc.page_content[:60]}...")

        return top_docs


class SimpleReranker:
    """简化版 Reranker：基于 LLM 打分"""

    def __init__(self):
        from model.factory import chat_model
        self.llm = chat_model

    def rerank(self, query: str, documents: list[Document], top_k: int = 3) -> list[Document]:
        from utils.logger_handler import logger

        scored_docs = []
        for doc in documents:
            prompt = f"""评估以下文档是否与用户问题相关。
只输出 0 到 10 之间的一个数字：
- 0 = 完全不相关
- 10 = 完全匹配

用户问题: {query}
文档内容: {doc.page_content[:200]}
相关性评分:"""

            try:
                response = self.llm.invoke([
                    {"role": "system", "content": "你是一个相关性评估专家，只输出数字。"},
                    {"role": "user", "content": prompt},
                ])
                score = float(response.content.strip())
            except Exception as e:
                logger.warning(f"LLM 评分失败，默认给 5 分: {e}")
                score = 5.0

            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored_docs[:top_k]]

        for i, (score, doc) in enumerate(scored_docs[:top_k]):
            print(f"  [{i+1}] score={score} | {doc.page_content[:60]}...")

        return top_docs
