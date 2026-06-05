"""Agent 模块测试"""


class TestReactAgent:
    def test_init(self):
        from agent.react_agent import ReactAgent
        agent = ReactAgent()
        assert agent is not None

    def test_execute_stream_yields_strings(self):
        from agent.react_agent import ReactAgent
        agent = ReactAgent()
        chunks = list(agent.execute_stream("你好"))
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, str)

    def test_tool_call_tracking(self):
        from agent.react_agent import ReactAgent
        agent = ReactAgent()
        list(agent.execute_stream("深圳天气怎么样？"))
        assert isinstance(agent.last_tool_calls, list)


class TestManualAgent:
    def test_init_default(self):
        from agent.manual_agent import ManualReactAgent
        agent = ManualReactAgent()
        assert agent is not None
        assert agent.max_iterations == 5

    def test_init_with_custom_prompt(self):
        from agent.manual_agent import ManualReactAgent
        agent = ManualReactAgent(
            system_prompt_path="prompts/orchestrator/customer_service_prompt.txt",
            max_iterations=3,
        )
        assert agent.max_iterations == 3


class TestMultiAgentSystem:
    def test_init(self):
        from agent.orchestrator import MultiAgentSystem
        system = MultiAgentSystem()
        assert system is not None
        assert system.orchestrator is not None
        assert system.customer_service is not None
        assert system.report is not None

    def test_orchestrator_classify(self):
        from agent.orchestrator import OrchestratorAgent
        orch = OrchestratorAgent()
        result = orch.classify("你好，我想咨询产品")
        assert "intent" in result
        assert "confidence" in result

    def test_customer_service_execute(self):
        from agent.orchestrator import CustomerServiceAgent
        agent = CustomerServiceAgent()
        result = agent.execute("小户型适合什么机器人？")
        assert isinstance(result, str)

    def test_report_agent_execute(self):
        from agent.orchestrator import ReportAgent
        agent = ReportAgent()
        result = agent.execute("帮我生成使用报告")
        assert isinstance(result, str)

    def test_execute_stream(self):
        from agent.orchestrator import MultiAgentSystem
        system = MultiAgentSystem()
        chunks = list(system.execute_stream("你好"))
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, str)
