"""
FastAPI 智能客服 API
提供流式(SSE)和非流式两种调用方式
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agent.react_agent import ReactAgent
from agent.orchestrator import MultiAgentSystem

app = FastAPI(title="智扫通智能客服 API", version="1.0.0")

_agent = None
_multi_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = ReactAgent()
    return _agent


def get_multi_agent():
    global _multi_agent
    if _multi_agent is None:
        _multi_agent = MultiAgentSystem()
    return _multi_agent


class ChatRequest(BaseModel):
    query: str
    multi_agent: bool = False


class ChatResponse(BaseModel):
    response: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.multi_agent:
        agent = get_multi_agent()
        result = agent.execute(req.query)
    else:
        agent = get_agent()
        result = ""
        for chunk in agent.execute_stream(req.query):
            result += chunk
    return ChatResponse(response=result.strip())


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    agent = get_agent()

    def generate():
        for chunk in agent.execute_stream(req.query):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
