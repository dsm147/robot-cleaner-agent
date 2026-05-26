"""
命令行版 Multi-Agent 智能客服
"""
from agent.orchestrator import MultiAgentSystem


def main():
    agent = MultiAgentSystem()

    print("="*60)
    print("智扫通 Multi-Agent 智能客服 (命令行版)")
    print("支持: 产品咨询 · 故障排查 · 使用报告")
    print("输入 'exit' 或 'quit' 退出")
    print("="*60)

    while True:
        user_input = input("\n你 > ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break

        print("\n智扫通 > ", end="", flush=True)
        response = agent.execute(user_input)
        print(response)
        print()


if __name__ == "__main__":
    main()
