"""
结构化输出：用 Pydantic 模型约束 LLM 输出格式
"""
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model


class RAGAnswer(BaseModel):
    """RAG 回答的结构化输出模型"""
    answer: str = Field(description="对用户问题的回答内容")
    confidence: float = Field(description="回答的置信度，0-1 之间的浮点数", ge=0, le=1)
    sources: list[str] = Field(description="回答引用的参考资料来源或主题")
    related_topics: list[str] = Field(
        description="用户可能感兴趣的相关话题",
        default_factory=list,
    )


parser = PydanticOutputParser(pydantic_object=RAGAnswer)

STRUCTURED_PROMPT = """你是专注于"基于参考资料总结"的AI助手。

### 输入信息
用户提问：{input}
参考资料：{context}

### 输出要求
{format_instructions}

### 约束
1. 事实准确：完全基于参考资料，不编造
2. 仅用中文回答
3. confidence 表示你对回答的确信程度
"""


def rag_summarize_structured(query: str, context: str) -> RAGAnswer:
    prompt = PromptTemplate(
        template=STRUCTURED_PROMPT,
        input_variables=["input", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    formatted_prompt = prompt.format(input=query, context=context)

    response = chat_model.invoke([
        {"role": "system", "content": "你是一个结构化的智能客服助手。"},
        {"role": "user", "content": formatted_prompt},
    ])

    try:
        structured_answer = parser.parse(response.content)
        return structured_answer
    except Exception as e:
        print(f"[结构化输出] 解析失败: {e}")
        return RAGAnswer(
            answer=response.content,
            confidence=0.5,
            sources=["解析失败"],
            related_topics=[],
        )


if __name__ == "__main__":
    result = rag_summarize_structured(
        query="X90 吸力多大？",
        context="X90 Pro Max 采用 5000Pa 大吸力设计。",
    )
    print(f"回答: {result.answer}")
    print(f"置信度: {result.confidence}")
    print(f"来源: {result.sources}")
    print(f"相关话题: {result.related_topics}")
