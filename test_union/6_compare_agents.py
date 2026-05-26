"""
对比测试：手写 Agent vs LangChain Agent
"""
import path_setup
import time
from agent.react_agent import ReactAgent
from agent.manual_agent import ManualReactAgent


def compare_agents():
    test_cases = [
        "小户型适合什么机器人？",
        "深圳今天天气怎么样？",
        "生成我的使用报告",
    ]

    agents = {
        "LangChain Agent": ReactAgent(),
        "手写 Agent": ManualReactAgent(),
    }

    for name, agent in agents.items():
        print(f"\n{'='*60}")
        print(f"测试: {name}")
        print(f"{'='*60}")

        for question in test_cases:
            print(f"\n>>> 用户: {question}")
            start = time.time()

            response = ""
            for chunk in agent.execute_stream(question):
                response += chunk

            elapsed = time.time() - start

            print(f">>> {name}: {response[:150]}...")
            print(f">>> 耗时: {elapsed:.2f}s")
            print(f">> 响应长度: {len(response)} 字")


if __name__ == "__main__":
    compare_agents()
