"""API 接口测试"""
from fastapi.testclient import TestClient


def test_health():
    from app.api_server import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat():
    from app.api_server import app
    client = TestClient(app)
    resp = client.post("/chat", json={"query": "你好"})
    assert resp.status_code == 200
    assert "response" in resp.json()


def test_chat_stream():
    from app.api_server import app
    client = TestClient(app)
    with client.stream("POST", "/chat/stream", json={"query": "你好"}) as resp:
        assert resp.status_code == 200


def test_chat_empty_query():
    from app.api_server import app
    client = TestClient(app)
    resp = client.post("/chat", json={"query": ""})
    assert resp.status_code == 200
