import { useState, useRef, useEffect } from "react";

interface Message {
  role: "user" | "assistant";
  content: string;
}

function App() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "你好！我是智扫通智能客服，有什么可以帮你的吗？" },
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSend() {
    const q = query.trim();
    if (!q || loading) return;

    setQuery("");
    setError(null);
    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setLoading(true);

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, multi_agent: false }),
      });

      if (!res.ok) {
        throw new Error(`请求失败 (${res.status})`);
      }

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.response || "（空回答）" },
      ]);
    } catch (e: any) {
      setError(e.message || "网络错误，请检查后端服务是否启动");
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>🤖 智扫通智能客服</h1>
        <p className="subtitle">扫地机器人 AI 客服 · 产品咨询 / 故障排查 / 使用报告</p>
      </header>

      <main className="chat">
        <div className="messages">
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <div className="avatar">{msg.role === "user" ? "👤" : "🤖"}</div>
              <div className="bubble">{msg.content}</div>
            </div>
          ))}

          {loading && (
            <div className="message assistant">
              <div className="avatar">🤖</div>
              <div className="bubble loading">
                <span className="dot" />
                <span className="dot" />
                <span className="dot" />
              </div>
            </div>
          )}

          {error && (
            <div className="message assistant">
              <div className="avatar">⚠️</div>
              <div className="bubble error">{error}</div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        <div className="input-bar">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="请输入您的问题..."
            disabled={loading}
          />
          <button onClick={handleSend} disabled={loading || !query.trim()}>
            {loading ? "..." : "发送"}
          </button>
        </div>
      </main>
    </div>
  );
}

export default App;
