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
            system_prompt_path="prompts/customer_service_prompt.txt",
            max_iterations=3,
        )
        assert agent.max_iterations == 3
