
"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复

支持的检索模式：
  - "vector"  纯向量检索
  - "hybrid"  混合检索（向量 + BM25）
  - "rewrite" Query Rewrite + 混合检索
  - "hyde"    HyDE（假设文档）+ 混合检索
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from rag.query_rewrite import rewrite_query
from rag.hyde import generate_hypothetical_document
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from utils.config_handler import chroma_conf


def print_prompt(prompt):
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt


class RagSummarizeService(object):
    def __init__(self, retriever_mode="hybrid", use_reranker=False, use_compressor=False, structured_output=False):
        """
        :param retriever_mode:
            "vector"     - 纯向量检索
            "hybrid"     - 混合检索（向量 + BM25）
            "rewrite"    - Query Rewrite + 混合检索
            "hyde"       - HyDE + 混合检索
        :param use_reranker: 是否使用 Cross-encoder Reranker 精排
        :param use_compressor: 是否使用 Context 压缩
        :param structured_output: 是否使用结构化输出
        """
        self.vector_store = VectorStoreService()
        self.retriever_mode = retriever_mode
        self.use_reranker = use_reranker
        self.use_compressor = use_compressor
        self.structured_output = structured_output

        # 检索时取更多结果留给 Reranker 精排
        search_k = chroma_conf["reranker_candidates"] if use_reranker else chroma_conf["k"]
        self.retriever = self.vector_store.get_hybrid_retriever(retriever_k=search_k)

        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

        # Reranker 和 Compressor
        self.reranker = None
        self.compressor = None
        if use_reranker:
            from rag.reranker import CrossEncoderReranker
            self.reranker = CrossEncoderReranker()
        if use_compressor:
            from rag.context_compressor import LLMContextCompressor
            self.compressor = LLMContextCompressor()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def _preprocess_query(self, query: str) -> str:
        """根据模式对查询进行预处理"""
        if self.retriever_mode == "rewrite":
            return rewrite_query(query)
        elif self.retriever_mode == "hyde":
            return generate_hypothetical_document(query)
        else:
            return query

    def retriever_docs(self, query: str) -> list[Document]:
        # 1. 查询预处理
        processed_query = self._preprocess_query(query)
        print(f"[检索模式: {self.retriever_mode}] 实际检索用查询: {processed_query[:100]}")

        # 2. 第一阶段检索
        docs = self.retriever.invoke(processed_query)
        print(f"[检索] 第一阶段获取 {len(docs)} 条候选")

        # 3. 第二阶段精排（Reranker）
        if self.use_reranker and self.reranker and len(docs) > chroma_conf["k"]:
            docs = self.reranker.rerank(query, docs, top_k=chroma_conf["k"])
            print(f"[检索] 第二阶段精排保留 {len(docs)} 条")

        # 4. Context 压缩
        if self.use_compressor and self.compressor and docs:
            docs = self.compressor.compress(query, docs)
            print(f"[检索] Context 压缩后保留 {len(docs)} 条")

        return docs

    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriever_docs(query)

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        if self.structured_output:
            from rag.structured_output import rag_summarize_structured
            result = rag_summarize_structured(query, context)
            # 工具返回给 LLM 的必须是字符串
            return (
                f"回答: {result.answer}\n"
                f"置信度: {result.confidence}\n"
                f"来源: {', '.join(result.sources)}"
            )

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == '__main__':
    rag = RagSummarizeService()
    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
