import time

import streamlit as st
from agent.react_agent import ReactAgent

# 页面配置
st.set_page_config(
    page_title="智扫通智能客服",
    page_icon="🤖",
    layout="centered",
)

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")
    if st.button("🗑️ 清空对话"):
        st.session_state["message"] = []
        st.rerun()

    st.divider()
    st.caption("**智扫通** — 扫地机器人 AI 客服")
    st.caption("支持产品咨询 · 故障排查 · 使用报告")
    st.caption("---")
    st.caption("模型: Qwen3-max (DashScope)")
    st.caption("RAG: ChromaDB + BM25 混合检索")

# 标题
st.title("🤖 智扫通机器人智能客服")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input("请输入您的问题...")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("智能客服思考中..."):
        res_stream = st.session_state["agent"].execute_stream(prompt)

        def capture(generator, cache_list):

            for chunk in generator:
                cache_list.append(chunk)

                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})
        st.rerun()
