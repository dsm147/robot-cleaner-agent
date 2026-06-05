"""
Query Rewrite：将用户口语化问题改写为适合检索的查询词
"""
from model.factory import get_chat_model


REWRITE_PROMPT = """你是一个检索查询优化专家。你的任务是将用户的自然语言问题改写为简洁、精确的检索查询，用于在扫地机器人知识库中搜索。

要求：
1. 提取核心关键词，去除口语化表达（如"这东西""那个"等）
2. 使用知识库可能使用的书面语表达
3. 直接输出查询文本，不需要多余解释

用户问题: {input}
检索查询:"""


def rewrite_query(user_input: str) -> str:
    """将用户问题改写为检索查询"""
    chat_model = get_chat_model()
    prompt = REWRITE_PROMPT.format(input=user_input)

    response = chat_model.invoke([
        {"role": "system", "content": "你是一个专业的检索查询改写助手。"},
        {"role": "user", "content": prompt},
    ])

    rewritten = response.content.strip()
    # 清理可能的引号
    rewritten = rewritten.strip('"').strip("'")
    print(f"[Query Rewrite] {user_input} → {rewritten}")
    return rewritten


if __name__ == "__main__":
    test_queries = [
        "这东西能自己拖地吗？",
        "噪音大不大？",
        "哪个适合养猫的家庭？",
        "电池能用多久？",
    ]
    for q in test_queries:
        print(f"原问题: {q}")
        print(f"改写后: {rewrite_query(q)}")
        print()
