"""
Context 压缩：对检索到的文档片段进行压缩，去除不相关的内容
"""
from typing import list
from langchain_core.documents import Document
from model.factory import chat_model


COMPRESS_PROMPT = """你是一个文档压缩专家。给定用户问题和一段参考资料，你的任务是：
1. 提取参考资料中与用户问题最相关的部分
2. 删除不相关的信息
3. 保持原文的表述不变，不要改写
4. 如果整段都不相关，返回"无相关信息"

用户问题: {query}
参考资料: {document}
压缩后的版本:"""


class LLMContextCompressor:
    """用 LLM 对每条文档做压缩"""

    def __init__(self):
        self.llm = chat_model

    def compress(self, query: str, documents: list[Document]) -> list[Document]:
        compressed_docs = []
        for doc in documents:
            prompt = COMPRESS_PROMPT.format(query=query, document=doc.page_content)

            response = self.llm.invoke([
                {"role": "system", "content": "你是一个文档压缩助手。"},
                {"role": "user", "content": prompt},
            ])

            compressed_content = response.content.strip()

            if compressed_content and compressed_content != "无相关信息":
                compressed_docs.append(Document(
                    page_content=compressed_content,
                    metadata=doc.metadata,
                ))

        print(f"[Context Compressor] 压缩前 {len(documents)} 条，压缩后 {len(compressed_docs)} 条")
        return compressed_docs


class LengthContextCompressor:
    """简单版本：按长度截断，去掉过长的文档"""

    def __init__(self, max_length: int = 300):
        self.max_length = max_length

    def compress(self, query: str, documents: list[Document]) -> list[Document]:
        compressed_docs = []
        for doc in documents:
            if len(doc.page_content) > self.max_length:
                content = doc.page_content[:self.max_length] + "..."
            else:
                content = doc.page_content

            compressed_docs.append(Document(
                page_content=content,
                metadata=doc.metadata,
            ))

        return compressed_docs
