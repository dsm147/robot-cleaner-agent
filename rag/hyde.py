"""
HyDE (Hypothetical Document Embeddings)：
先让 LLM 基于问题生成假设答案，再用假设答案做向量检索
"""
from model.factory import chat_model


HYDE_PROMPT = """你是一个扫地机器人专家。请基于用户的问题，生成一段详细、专业的回答。

要求：
1. 回答要具体，包含型号名、技术参数等细节（可以合理推测）
2. 回答风格要和产品知识库文档一致：客观、技术性
3. 长度在 100-200 字之间
4. 直接输出回答内容，不需要多余解释

用户问题: {input}
专业回答:"""


def generate_hypothetical_document(user_input: str) -> str:
    """生成假设文档用于检索"""
    prompt = HYDE_PROMPT.format(input=user_input)

    response = chat_model.invoke([
        {"role": "system", "content": "你是一个扫地机器人产品专家。"},
        {"role": "user", "content": prompt},
    ])

    hypo_doc = response.content.strip()
    print(f"[HyDE] 假设文档生成完成 ({len(hypo_doc)} 字)")
    return hypo_doc


if __name__ == "__main__":
    test_queries = [
        "扫地机器人能把瓜子壳吸干净吗？",
        "小户型适合什么机器人？",
    ]
    for q in test_queries:
        print(f"问题: {q}")
        print(f"假设文档: {generate_hypothetical_document(q)}")
        print()
