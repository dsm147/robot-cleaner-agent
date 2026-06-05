"""
FastAPI 智能客服 API
提供流式(SSE)和非流式两种调用方式
"""
import sys
import os
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent.react_agent import ReactAgent
from agent.orchestrator import MultiAgentSystem


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时校验关键配置"""
    if not os.environ.get("DASHSCOPE_API_KEY"):
        import warnings
        warnings.warn(
            "⚠️ DASHSCOPE_API_KEY 未设置！请通过环境变量配置后再调用 API。"
        )
    yield


app = FastAPI(title="智扫通智能客服 API", version="1.0.0", lifespan=lifespan)

# CORS 配置：允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# 提供前端静态文件服务（生产环境）
frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.isdir(frontend_dist):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
