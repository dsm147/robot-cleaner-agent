# 智扫通机器人智能客服系统

![Python](https://img.shields.io/badge/python-3.11-blue)
[![CI](https://github.com/dsm147/agent-rag/actions/workflows/test.yml/badge.svg)](https://github.com/dsm147/agent-rag/actions/workflows/test.yml)

一个面向扫地机器人 / 扫拖一体机器人的 AI 智能客服系统，结合 **RAG（检索增强生成）** 与 **AI Agent（智能体）** 技术，提供产品咨询、故障排查、使用报告生成等功能。

---

## 项目结构

```
├── agent/                   # Agent 层
│   ├── react_agent.py       # LangChain ReAct Agent（推荐，生产用）
│   ├── manual_agent.py      # 手写 ReAct Agent（学习/调试用）
│   ├── orchestrator.py      # Multi-Agent 调度器
│   └── tools/
│       ├── agent_tools.py   # 7 个工具定义
│       └── middleware.py    # Agent 中间件（日志/提示词切换）
├── rag/                     # RAG 检索层
│   ├── rag_service.py       # RAG 总结服务（检索+生成）
│   ├── vector_store.py      # ChromaDB 向量库 + BM25 混合检索
│   ├── query_rewrite.py     # 查询改写
│   ├── hyde.py              # 假设文档检索
│   ├── reranker.py          # Cross-encoder 重排序
│   ├── context_compressor.py# 上下文压缩
│   └── structured_output.py # 结构化输出
├── model/
│   └── factory.py           # 模型工厂（懒加载单例）
├── utils/
│   ├── config_handler.py    # YAML 配置加载
│   ├── path_tool.py         # 路径工具
│   ├── prompt_loader.py     # 提示词文件加载
│   ├── logger_handler.py    # 日志配置
│   └── file_handler.py      # 文件处理
├── config/                  # YAML 配置
│   ├── rag.yml              # 模型/检索模式配置
│   ├── chroma.yml           # 向量库参数
│   ├── agent.yml            # Agent 数据路径
│   └── prompts.yml          # 提示词文件路径
├── prompts/                 # 系统提示词
├── data/                    # 知识库源文件
│   └── external/records.csv # 模拟用户使用记录
├── eval/                    # 评估系统
│   ├── eval_rag.py          # RAG 质量评估
│   ├── eval_agent.py        # Agent 行为评估
│   ├── eval_metrics.py      # 量化指标（Hit Rate, MRR）
│   └── test_dataset.json    # 测试用例
├── tests/                   # pytest 自动化测试
│   ├── conftest.py          # 共享 Mock 配置
│   ├── test_rag.py          # RAG 测试
│   ├── test_agent.py        # Agent + Multi-Agent 测试
│   ├── test_api.py          # API 测试
│   └── test_utils.py        # 工具函数测试
├── test_union/              # 手动集成测试
├── app.py                   # Streamlit Web 界面
├── api_server.py            # FastAPI REST API（生产环境也提供前端静态文件）
├── cli_multi_agent.py       # 命令行多 Agent
├── frontend/                # React 聊天界面 (Vite + TypeScript)
│   ├── src/App.tsx          #   聊天组件
│   └── vite.config.ts       #   开发环境代理 /chat → localhost:8000
├── Dockerfile               # 多阶段构建
├── docker-compose.yml       # 生产部署
├── docker-compose.override.yml # 本地开发覆盖
├── .env.example             # 环境变量示例
└── requirements.txt         # Python 依赖
```

---

## 技术栈

| 层 | 技术 |
|---|---|
| 框架 | FastAPI (API) / Streamlit (UI) |
| LLM | Qwen3-max (通义千问) via DashScope API |
| 向量库 | ChromaDB + text-embedding-v4 |
| 检索 | 混合检索（向量 + BM25）+ Cross-encoder Reranker |
| Agent | LangChain Agent / 手写 ReAct / LangGraph 三种实现 |
| 评估 | Hit Rate, MRR, 工具调用准确率 |
| 测试 | pytest（单元 + 集成）|
| 部署 | Docker 多阶段构建 + docker-compose / GitHub Actions CI |

---

## 食用指南：怎么读懂这个项目

这个项目分为 **5 个逻辑层次**，从下往上依次是：

```
┌──────────────────────────────────────────┐
│  第五层：应用层 (app.py / cli*.py)        │
├──────────────────────────────────────────┤
│  第四层：Agent 层 (react/manual/orch)     │
├──────────────────────────────────────────┤
│  第三层：RAG 检索层 (vector/hybrid/rerank)│
├──────────────────────────────────────────┤
│  第二层：模型与工具层 (model/utils)        │
├──────────────────────────────────────────┤
│  第一层：配置与数据层 (config/data/prompts)│
└──────────────────────────────────────────┘
```

建议按 **从底层到顶层** 的顺序阅读，先搞清楚数据从哪来、怎么存，再看检索、Agent、应用。

---

## 第一层：配置与数据（地基）

### `config/` — 所有配置文件

| 文件 | 何时用到 | 作用 |
|---|---|---|
| **`rag.yml`** | 修改模型/检索模式时 | 指定聊天模型（如 qwen3-max）、嵌入模型、默认检索模式 |
| **`chroma.yml`** | 调整向量库参数时 | 向量库名、分块大小（chunk_size）、检索数量（k）、数据目录等 |
| **`agent.yml`** | Agent 读取外部数据时 | 外部数据文件路径（external_data_path） |
| **`prompts.yml`** | 修改提示词文件位置时 | 各提示词文件的相对路径 |

> **关联文件**: `config_handler.py`（第 2 层）读取这些 yml 文件。

### `data/` — 知识库源数据

这是 RAG 系统的**原始知识来源**：

| 文件 | 何时用到 | 作用 |
|---|---|---|
| **`data/扫地机器人100问.pdf`** | 加载知识库时 | PDF 格式的产品知识 |
| **`data/扫地机器人100问2.txt`** | 同上 | 更多问答对 |
| **`data/扫拖一体机器人100问.txt`** | 同上 | 扫拖一体机专属知识 |
| **`data/故障排除.txt`** | 同上 | 故障代码处理方法 |
| **`data/维护保养.txt`** | 同上 | 维护保养知识 |
| **`data/选购指南.txt`** | 同上 | 选购建议知识 |
| **`data/external/records.csv`** | 生成使用报告时 | 模拟用户使用记录数据（供 `fetch_external_data` 工具读取） |

> 这些文件通过 `vector_store.py` 的 `load_document()` 读入，分块后存入 ChromaDB 向量库。
> `chroma.yml` 中的 `data_path` 和 `allow_knowledge_file_type` 控制读取哪些文件。

### `prompts/` — 提示词文件（LLM 的"剧本"）

| 文件 | 谁在用 | 作用 |
|---|---|---|
| **`main_prompt.txt`** | `react_agent.py` / `manual_agent.py`（默认） | 主系统提示词：定义 Agent 的 ReAct 思考流程、7 个工具的使用规则 |
| **`rag_summarize.txt`** | `rag_service.py` 的 `RagSummarizeService` | RAG 总结提示词：告诉 LLM 如何基于检索到的参考资料回答问题 |
| **`orchestrator_prompt.txt`** | `orchestrator.py` 的 `OrchestratorAgent` | 调度提示词：判断用户意图是"客服咨询"还是"报告生成" |
| **`customer_service_prompt.txt`** | `orchestrator.py` 的 `CustomerServiceAgent` | 客服 Agent 提示词：简化版，只有 3 个工具 |
| **`report_agent_prompt.txt`** | `orchestrator.py` 的 `ReportAgent` | 报告 Agent 提示词：只有报告相关工具 |
| **`report_switch_prompt.txt`** | `middleware.py` 的动态提示词切换 | 报告场景的备用提示词：当 `fill_context_for_report` 被调用后替换主提示词 |

> **提示词关系图**：
> - 单 Agent 模式：`main_prompt.txt` 包含全部 7 个工具，Agent 自主判断用哪个
> - 多 Agent 模式：`orchestrator_prompt.txt` 做意图分类 → 分派给 `customer_service_prompt.txt`（3 工具）或 `report_agent_prompt.txt`（4 工具）

---

## 第二层：模型与工具（基础设施）

### `model/factory.py` — 模型工厂

**何时用到**：几乎每次 LLM 调用都会经过这里。

**功能**：
- 从 `rag.yml` 读取模型名称（默认 `qwen3-max`）
- 从环境变量 `DASHSCOPE_API_KEY` 读取 API Key
- 懒加载单例：`chat_model` 和 `embed_model` 在首次访问时初始化，不在 import 时初始化

**关联文件**：
- ← 读取 `config/rag.yml`
- → 被 `agent/*.py`、`rag/*.py` 等几乎所有模块导入

### `utils/` — 工具函数

| 文件 | 何时用到 | 作用 | 被谁调用 |
|---|---|---|---|
| **`path_tool.py`** | 任何时候需要读写文件 | 获取项目根目录绝对路径，所有文件操作都用它 | 几乎全部模块 |
| **`config_handler.py`** | 启动时读取配置 | 加载 yml 配置文件，暴露 `rag_conf`、`chroma_conf`、`prompts_conf`、`agent_conf` | `factory.py`、`vector_store.py`、`rag_service.py`、`prompt_loader.py`、`agent_tools.py` |
| **`prompt_loader.py`** | 加载提示词时 | 读取 `prompts.yml` 中配置的提示词文件内容 | `react_agent.py`、`manual_agent.py`、`middleware.py`、`rag_service.py` |
| **`logger_handler.py`** | 需要记日志时 | 配置日志格式，输出到控制台 + `logs/` 目录 | `vector_store.py`、`middleware.py`、`agent_tools.py`、`manual_agent.py`、`file_handler.py` |
| **`file_handler.py`** | 读写知识库文件时 | PDF/TXT 文件加载、MD5 计算、目录文件过滤 | `vector_store.py` |

---

## 第三层：RAG 检索层（核心能力）

这是整个系统的**知识检索中台**。用户提问后，RAG 负责从知识库中找到最相关的资料。

### `rag/vector_store.py` — 向量库服务

**何时用到**：只要需要检索知识库就会用到。

**功能**：
- 使用 **ChromaDB** 作为向量数据库
- 提供「纯向量检索」和「混合检索（向量 + BM25 关键词）」两种方式
- `load_document()`：读取 `data/` 下的知识文件 → 分片（`RecursiveCharacterTextSplitter`）→ 向量化 → 存入 ChromaDB
  - 通过 **MD5 去重**避免重复加载（md5 记录在 `chroma.yml` 的 `md5_hex_store` 字段指向的文件）
- `get_hybrid_retriever()`：创建混合检索器，组合向量 + BM25，权重各 0.5
- BM25 有**缓存机制**（`bm25_cache.pkl` + `bm25_cache_key.txt`），源文件没变就不重建

**关联**：
- ← 使用 `file_handler.py` 加载文件、计算 MD5
- ← 使用 `model/factory.py` 的 `embed_model` 做向量化
- ← 读取 `config/chroma.yml`
- → 被 `rag_service.py` 调用

### `rag/rag_service.py` — RAG 总结服务

**何时用到**：用户提问后，需要检索 + 生成回答时。

**功能**（`RagSummarizeService`）：
1. 接收用户问题
2. 根据 `retriever_mode` 对查询做**预处理**（见下方 4 种模式）
3. 用预处理后的查询做**检索**
4. 可选：**Reranker 精排** / **Context 压缩**
5. 将检索结果拼入提示词，调用 LLM **生成回答**

**4 种检索模式**（通过 `retriever_mode` 参数切换）：

| 模式 | 做了什么 | 适用场景 |
|---|---|---|
| `vector` | 直接用原始问题做向量检索 | 基线、快速测试 |
| `hybrid` | 向量 + BM25 混合检索 | 大多数场景，默认推荐 |
| `rewrite` | 先把口语问题改写为书面查询，再做混合检索 | 用户提问口语化、模糊时 |
| `hyde` | 先让 LLM 生成假设文档，再用假设文档检索 | 查询与文档措辞差异大时 |

**可选增强链路**（通过构造参数控制）：

```
原始问题 → [Query Rewrite / HyDE] → 混合检索 → [Reranker] → [Context Compressor] → [结构化输出] → LLM 生成
```

**关联**：
- ← 调用 `vector_store.py` 做检索
- ← 调用 `query_rewrite.py` / `hyde.py` / `reranker.py` / `context_compressor.py` / `structured_output.py`
- ← 读取 `config/chroma.yml`、`config/rag.yml`
- ← 使用 `prompt_loader.py` 加载 `rag_summarize.txt`
- → 被 `agent_tools.py` 的 `rag_summarize` 工具调用
- → 被测试文件直接调用

### `rag/query_rewrite.py` — 查询改写

**何时用到**：`retriever_mode="rewrite"` 时。

**功能**：用 LLM 将用户的口语化问题（如"这东西能自己拖地吗？"）改写为适合检索的书面查询（如"扫拖一体机器人 自动拖地 功能"）。

### `rag/hyde.py` — 假设文档检索

**何时用到**：`retriever_mode="hyde"` 时。

**功能**：先让 LLM 基于问题生成一段假设性的专业回答，再用这段回答做检索。当用户提问方式和知识库文档风格差异很大时有效。

### `rag/reranker.py` — 重排序

**何时用到**：`use_reranker=True` 时。

**功能**：检索出候选文档后，用 **Cross-encoder** 模型（`BAAI/bge-reranker-v2-m3`）对文档与问题的相关性做二次打分排序，提升前排文档的质量。

**两个实现**：
- `CrossEncoderReranker`：专用 Cross-encoder 模型，精度高
- `SimpleReranker`：用 LLM 逐条打分，不需要额外模型

### `rag/context_compressor.py` — 上下文压缩

**何时用到**：`use_compressor=True` 时。

**功能**：用 LLM 对每条文档做压缩，只保留与问题最相关的部分，减少上下文长度。

**两个实现**：
- `LLMContextCompressor`：用 LLM 智能压缩
- `LengthContextCompressor`：按长度截断

### `rag/structured_output.py` — 结构化输出

**何时用到**：`structured_output=True` 时。

**功能**：约束 LLM 输出为结构化的 Pydantic 模型（`RAGAnswer`），包含 answer、confidence、sources、related_topics 字段。

---

## 第四层：Agent 层（智能体）

这一层让 RAG"活"起来——Agent 能自主决定何时检索、何时查天气、何时生成报告，支持多轮工具调用。

### `agent/tools/agent_tools.py` — 工具定义

**何时用到**：所有 Agent 模式都需要。

**功能**：定义 7 个工具（用 `@tool` 装饰器）：

| 工具 | 做什么 | 场景 |
|---|---|---|
| `rag_summarize(query)` | 从知识库检索回答 | 产品咨询、故障排查 |
| `get_weather(city)` | 获取天气（模拟数据） | 天气适配咨询 |
| `get_user_location()` | 获取用户城市（模拟随机） | 需要城市信息时 |
| `get_user_id()` | 获取用户 ID（模拟随机） | 生成报告时 |
| `get_current_month()` | 获取当前月份（模拟随机） | 生成报告时 |
| `fetch_external_data(user_id, month)` | 读取 CSV 使用记录 | 生成报告时 |
| `fill_context_for_report()` | 触发提示词切换标记 | 生成报告前必须调用 |

**关联**：
- → 调用 `rag_service.py` 做 RAG 检索
- → 读取 `config/agent.yml` 获取外部数据路径

### `agent/tools/middleware.py` — Agent 中间件

**何时用到**：使用 `react_agent.py`（LangChain Agent）时。

**功能**：三个中间件函数：
- `monitor_tool`：记录工具调用日志，检测 `fill_context_for_report` 调用并设置 context 标记
- `log_before_model`：在每次 LLM 调用前记录日志
- `report_prompt_switch`：动态提示词切换——检测到 report 场景时自动切换提示词

**关联**：
- → 调用 `prompt_loader.py` 加载提示词
- → 被 `react_agent.py` 装配到 Agent 中

### `agent/react_agent.py` — LangChain Agent（推荐）

**何时用到**：启动智能客服主界面（`app.py`）时。

**功能**：使用 LangChain 框架的 `create_agent` 创建 ReAct Agent。装配了全部 7 个工具 + 3 个中间件，通过 `execute_stream()` 提供流式输出。

**特点**：框架维护，稳定可靠，适合生产使用。

**关联**：
- ← 使用 `model/factory.py` 的 `chat_model`
- ← 使用 `prompt_loader.py` 加载 `main_prompt.txt`
- → 被 `app.py`（Streamlit UI）调用

### `agent/manual_agent.py` — 手写 Agent（学习用）

**何时用到**：学习 Agent 原理、调试工具调用时。

**功能**：不用 LangChain 框架，从零实现 ReAct 循环：
1. 构建消息列表（含历史对话）
2. 调用 LLM（传工具定义）
3. 解析响应中的 tool_calls
4. 执行工具，将结果追加到消息列表
5. 循环直到 LLM 返回纯文本回答
6. 包含**死循环检测**、**重试机制**、**对话历史管理**

**特点**：没有框架抽象，每一步清晰可见，方便调试和理解 Agent 原理。

**两个构造函数**：
- 无参（默认）：加载 `main_prompt.txt`，7 个工具全开
- 传 `system_prompt_path`：加载指定提示词（如只含部分工具），用于多 Agent 场景

**关联**：
- → 被 `graph_agent.py` 导入（复用 `TOOL_DEFINITIONS`、`TOOL_REGISTRY`）
- → 被 `orchestrator.py` 导入（作为 CustomerServiceAgent 和 ReportAgent 的基础）

### `agent/orchestrator.py` — 多 Agent 调度器

**何时用到**：需要多 Agent 分工时（通过 `cli_multi_agent.py`）。

**功能**：Orchestrator + Worker 架构：
1. **OrchestratorAgent**：用 `orchestrator_prompt.txt` 分析用户意图 → 返回 `{"intent": "customer_service" / "report"}`
2. **CustomerServiceAgent**：用 `ManualReactAgent(system_prompt_path="customer_service_prompt.txt")`，只有 3 个工具（rag_summarize、get_weather、get_user_location）
3. **ReportAgent**：用 `ManualReactAgent(system_prompt_path="report_agent_prompt.txt")`，只有 4 个报告相关工具

**关联**：
- ← 使用 `manual_agent.py` 作为 Worker
- → 被 `cli_multi_agent.py` 调用

---

## 第五层：应用层（用户界面）

### `app.py` — Streamlit Web 界面

**何时用到**：想通过网页和 AI 客服对话时。

**如何启动**：
```bash
streamlit run app.py
```

**功能**：Streamlit 聊天界面，使用 `ReactAgent`，支持流式输出、清空对话、侧边栏信息展示。

### `api_server.py` — FastAPI REST API（含前端静态服务）

**何时用到**：想通过 HTTP 调用智能客服时（集成到其他系统、小程序、企业微信等），或想直接使用 Web 聊天界面时。

**如何启动**：
```bash
python api_server.py
# 或: uvicorn api_server:app --host 0.0.0.0 --port 8000
```

**功能**：提供三个 API 接口 + 一个 Web 界面：
- `GET /health` — 健康检查
- `POST /chat` — 非流式聊天
- `POST /chat/stream` — SSE 流式聊天
- `GET /` — React 聊天界面（生产环境，需先构建前端）

**改进**：支持 CORS 跨域访问、启动时校验 API Key 配置。

**自动文档**：启动后访问 `http://localhost:8000/docs` 查看 Swagger 文档。

**React 前端**：生产环境下直接访问 `http://localhost:8000` 即可看到聊天界面（前端静态文件由 FastAPI 自动服务）。

### `frontend/` — React 聊天界面

**何时用到**：想通过现代 Web 界面和 AI 客服对话时。

**技术栈**：React 19 + TypeScript + Vite。

**开发模式启动**（前后端分离，支持热重载）：
```bash
# 终端 1：启动后端 API
python api_server.py

# 终端 2：启动前端开发服务器
cd frontend
npm install   # 首次运行需要
npm run dev   # 访问 http://localhost:5173
```

**生产模式**：`api_server.py` 在检测到 `frontend/dist/` 目录存在时，自动挂载为静态文件服务。Docker 构建会自动生产前端产物，部署后直接访问 `http://localhost:8000`。

### `cli_multi_agent.py` — 命令行多 Agent 界面

**何时用到**：想在终端体验多 Agent 分工时。

**如何启动**：
```bash
python cli_multi_agent.py
```

**功能**：命令行交互，使用 `MultiAgentSystem`（orchestrator.py），会打印意图分类结果和 Agent 分工过程。

---

## 评估层：评测系统

### `eval/eval_rag.py` — RAG 质量评估

**测试何时用到**：想评估不同检索模式的回答质量时。

**功能**：用 `test_dataset.json` 中的测试用例，运行不同配置（vector/hybrid/rewrite + reranker），记录响应时间和回答。

### `eval/eval_agent.py` — Agent 行为评估

**测试何时用到**：想评估 Agent 的工具调用准确率时。

**功能**：评估 Agent 的工具调用准确率（实际调用的工具是否匹配预期）和关键词命中率。

### `eval/compare_mono_vs_multi.py` — 单/多 Agent 对比

**测试何时用到**：想对比单 Agent 和多 Agent 的差异时。

**功能**：在同一批测试用例上运行 `ReactAgent` 和 `MultiAgentSystem`，对比响应时间和回答长度。

### `eval/run_all_eval.py` — 一键跑全部评估

**测试何时用到**：想一次性跑完所有评估时。

**功能**：依次运行 RAG 评估和 Agent 评估。

### `eval/test_dataset.json` — 测试数据集

**内容**：14 个测试用例，覆盖 6 个类别：
- 产品规格（spec）、故障处理（fault）、选购建议（buy）
- 天气适配（weather）、报告生成（report）、边界情况（edge）

每个用例包含 question、expected_tools、expected_answer、category。

### `eval/eval_metrics.py` — 量化评估指标

**何时用到**：想用数据说话时。

**功能**：计算检索质量的 **Hit Rate**（命中率）和 **MRR**（平均倒数排名），可选基于 RAGAS 框架的 **Faithfulness**（忠实度）和 **Answer Relevancy**（答案相关度）。

```bash
python eval/eval_metrics.py
```

---

## 自动化测试

### `tests/` — pytest 单元测试

使用 pytest 编写的自动化测试，Mock 了 LLM 调用，不消耗 API 费用。

```bash
# 安装测试依赖
pip install pytest httpx

# 运行全部测试
pytest tests/ -v

# 查看覆盖率
pip install pytest-cov
pytest tests/ --cov=. --cov-report=term-missing
```

| 文件 | 测试内容 |
|---|---|
| **`conftest.py`** | 共享 Mock 配置，自动 Mock `chat_model` |
| **`test_rag.py`** | RAG 初始化、检索模式切换、总结功能 |
| **`test_agent.py`** | ReactAgent / ManualAgent / MultiAgentSystem |
| **`test_api.py`** | API 健康检查、聊天接口、流式接口、边界情况 |
| **`test_utils.py`** | 路径工具、配置文件加载 |

## 手动测试脚本

### `test_union/` — 测试脚本集合

所有文件放在同一目录，共享 `path_setup.py` 的路径初始化。

| 文件 | 用途 | 什么时候用 |
|---|---|---|
| **`path_setup.py`** | 路径初始化，让 `from rag.*`、`from agent.*` 等导入正常工作 | 所有测试文件自动导入 |
| **`1_Tool_Calling_test.py`** | 直接调用 LLM，观察 Tool Calling 的请求/响应格式 | 想理解 Tool Calling 机制时 |
| **`2_two_retriever_mode_test.py`** | 对比 vector vs hybrid 的完整问答质量 | 想评估不同检索模式的回答效果时 |
| **`2_two_retrivers_res_test.py`** | 对比纯向量 vs 混合检索的原始文档差异（不含 LLM 生成） | 想知道检索器本身返回什么文档时 |
| **`3_retrieval_modes_test.py`** | 四种检索模式（vector/hybrid/rewrite/hyde）横向对比 | 想全面比较各检索模式时 |
| **`5_test_prompts.py`** | 测试当前 RAG 提示词版本的效果 | 修改了 `rag_summarize.txt` 后想测试效果时 |
| **`6_compare_agents.py`** | 对比 LangChain Agent vs 手写 Agent 的响应速度和效果 | 想评估两种 Agent 实现的差异时 |
| **`7_graph_agent.py`** | LangGraph 版单 Agent 实现 | 想学习或测试 LangGraph 框架时 |
| **`8_graph_multi_agent.py`** | LangGraph 版多 Agent 实现 | 想学习或测试 LangGraph 多 Agent 时 |

**运行方式**：
```bash
# 在项目根目录下运行
python test_union/1_Tool_Calling_test.py
python test_union/5_test_prompts.py
# 以此类推
```

---

## 学习教程

### `tutorial/` — 从零到一学本项目

**基础篇**（lesson_01 ~ lesson_10）：读懂项目代码，理解 RAG 和 Agent 原理。
**进阶篇**（lesson_11 ~ lesson_15）：把项目做到可部署上线，写进简历。

| 课程 | 文件 | 覆盖内容 |
|---|---|---|
| 第 1 课 | `lesson_01_project_anatomy.md` | 项目结构全览 |
| 第 2 课 | `lesson_02_hybrid_retrieval.md` | 混合检索原理 |
| 第 3 课 | `lesson_03_query_rewrite_hyde.md` | Query Rewrite 与 HyDE |
| 第 4 课 | `lesson_04_reranker_context_compression.md` | Reranker 与上下文压缩 |
| 第 5 课 | `lesson_05_prompt_engineering.md` | 提示词工程 |
| 第 6 课 | `lesson_06_manual_react_agent.md` | 手写 ReAct Agent |
| 第 7 课 | `lesson_07_memory_error_handling.md` | 记忆与错误处理 |
| 第 8 课 | `lesson_08_langgraph.md` | LangGraph 框架 |
| 第 9 课 | `lesson_09_evaluation.md` | 评估系统 |
| 第 10 课 | `lesson_10_multi_agent.md` | 多 Agent 系统 |
| 第 11 课 | `lesson_11_fastapi.md` | FastAPI + 流式 API |
| 第 12 课 | `lesson_12_testing.md` | pytest 自动化测试 |
| 第 13 课 | `lesson_13_docker.md` | Docker 部署 |
| 第 14 课 | `lesson_14_eval_metrics.md` | RAG 量化评估指标 |
| 第 15 课 | `lesson_15_cicd_showcase.md` | CI/CD + 简历写法 |

---

## 快速开始

### 前置条件

1. 注册阿里云百炼平台，获取 API Key
2. 开通 DashScope 服务（通义千问 + 文本嵌入）

### 配置环境变量

```bash
# Linux / macOS
export DASHSCOPE_API_KEY=your_api_key_here

# Windows PowerShell
$env:DASHSCOPE_API_KEY="your_api_key_here"

# Windows CMD
set DASHSCOPE_API_KEY=your_api_key_here
```

或复制 `.env.example` 为 `.env` 文件并填入密钥（需自行解析 .env 的工具）。

### 加载知识库

首次使用需要将 `data/` 下的知识文件加载到 ChromaDB：

```python
# 运行一次即可
from rag.vector_store import VectorStoreService
vs = VectorStoreService()
vs.load_document()
```

### 启动 Web 界面

```bash
streamlit run app.py
```

### 启动命令行多 Agent

```bash
python cli_multi_agent.py
```

### 启动 API 服务

```bash
python api_server.py
# 访问 http://localhost:8000/docs 查看 API 文档
```

### 启动测试

```bash
# 自动化测试
pytest tests/ -v

# 手动测试
python test_union/3_retrieval_modes_test.py
```

### Docker 部署

```bash
# 确保设置了 API Key
export DASHSCOPE_API_KEY=your_key_here

# 生产环境启动
docker compose up -d

# 开发环境启动（自动热重载）
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d

# 查看日志
docker compose logs -f
```

---

## 架构全景图（文字版）

```
用户输入
    │
    ▼
┌─────────────────────────────────────────────────┐
│  app.py (Streamlit) / cli_multi_agent.py (CLI)   │
│         │                                        │
│         ▼                                        │
│  ReactAgent / MultiAgentSystem                   │
│         │                                        │
│    ┌────┴────┐                                   │
│    │         │                                   │
│  Orchestrator  ManualReactAgent / LangChain Agent│
│  (意图分类)   │    │                             │
│    │         │    │ 中间件: monitor, log,        │
│    ▼         │    │        prompt_switch         │
│  ReportAgent │    ▼                              │
│  CS Agent    │  Tools (7个)                      │
│              │    │  ┌─────────────────┐         │
│              │    └──│ rag_summarize   │         │
│              │       │ → RagSummarizeService     │
│              │       │   → VectorStoreService    │
│              │       │   → ChromaDB + BM25       │
│              │       │   → [Reranker]            │
│              │       │   → [Compressor]          │
│              │       │   → [Structured Output]   │
│              │       │   → LLM 生成回答          │
│              │       └─────────────────┘         │
│              │       ┌─────────────────┐         │
│              │       │ get_weather     │ (模拟)   │
│              │       │ get_user_id     │ (模拟)   │
│              │       │ get_user_loc    │ (模拟)   │
│              │       │ get_month       │ (模拟)   │
│              │       │ fetch_ext_data  │ ← CSV   │
│              │       │ fill_ctx_report │ (标记)   │
│              │       └─────────────────┘         │
└─────────────────────────────────────────────────┘
```

---

## 文件依赖关系速查

```
你想问"改了这个文件会影响谁？"

config/*.yml
  → config_handler.py → 全部模块

prompts/*.txt
  → prompt_loader.py → react_agent / manual_agent / middleware / rag_service

data/ 知识文件
  → vector_store.py → rag_service.py → agent_tools.py → 所有 Agent

rag/vector_store.py
  → rag/rag_service.py → agent/tools/agent_tools.py → 所有 Agent

rag/query_rewrite.py, rag/hyde.py, rag/reranker.py, rag/context_compressor.py
  → rag/rag_service.py

rag/structured_output.py
  → rag/rag_service.py (可选)

model/factory.py
  → 全部需要 LLM 或 Embedding 的模块

agent/tools/agent_tools.py
  → agent/manual_agent.py → agent/orchestrator.py → cli_multi_agent.py
  → agent/react_agent.py → app.py

agent/tools/middleware.py
  → agent/react_agent.py → app.py

agent/manual_agent.py
  → agent/orchestrator.py → cli_multi_agent.py

agent/react_agent.py
  → app.py
  → api_server.py
  → eval/eval_agent.py
  → eval/compare_mono_vs_multi.py

agent/orchestrator.py
  → cli_multi_agent.py
  → api_server.py (可选 multi_agent=True)

api_server.py
  → tests/test_api.py

tests/conftest.py
  → tests/test_rag.py / test_agent.py / test_api.py / test_utils.py
```

---

## 补充说明

### 模拟数据

- `get_weather`、`get_user_id`、`get_user_location`、`get_current_month` 返回的都是模拟/随机数据，不代表真实值
- `data/external/records.csv` 包含模拟的用户使用记录数据
- `fetch_external_data` 读取该 CSV 返回数据

### 缓存文件

- `chroma_db/`：ChromaDB 向量库持久化目录，首次 `load_document()` 后生成
- `bm25_cache.pkl` + `bm25_cache_key.txt`：BM25 检索器缓存，源文件不变就不重建
- 以上已在 `.gitignore` 中排除，不会提交到 Git

### 日志

- 日志输出到控制台 + `logs/agent_YYYYMMDD.log`
- 由 `utils/logger_handler.py` 控制格式和级别
