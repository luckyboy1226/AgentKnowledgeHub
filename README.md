# 🤖 AgentKnowledgeHub — 企业级多Agent知识管理系统

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)
![Vue](https://img.shields.io/badge/Vue-3.x-4FC08D?logo=vue.js)
![LangGraph](https://img.shields.io/badge/LangGraph-0.3%2B-FF6B6B)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)

**一个企业级的「多Agent协作」知识管理系统**

4个AI Agent分工协作，完成企业知识的全生命周期管理：文档解析 → 知识抽取 → 智能问答 → 增量更新

[快速开始](#-快速开始) · [系统架构](#-系统架构) · [功能演示](#-功能演示) · [API文档](#-api-接口) 

</div>

---

## 📌 先看这里

> 如果你是第一次接触多Agent系统，先看这几个问题的解答：

### 什么是 Agent？
Agent（智能体）就是一个"能思考、能执行"的AI程序。它可以：
- 理解你的需求（自然语言）
- 决定需要调用哪些工具（如搜索、写文件、调用API）
- 执行工具，得到结果
- 根据结果继续思考，直到完成任务

### 什么是多Agent？
当一个任务太复杂，交给多个专职Agent协作完成。就像公司里：
- **秘书** 负责整理文件
- **分析师** 负责提炼关键信息
- **顾问** 负责回答问题
- **管理员** 负责持续更新维护

本项目就是用AI实现了这4个角色的分工协作。

### 这个项目能做什么？
你上传一份公司的PDF文档（比如年报、合同、产品手册），然后可以：
- 直接用自然语言提问："张三的职位是什么？" / "Q3营收多少？"
- AI会综合理解文档内容，给出准确答案
- 文档更新后，知识库自动同步，不用重新上传

---

## 📋 目录

- [项目简介](#-项目简介)
- [系统架构](#-系统架构)
- [技术栈](#-技术栈)
- [快速开始](#-快速开始)
- [功能演示](#-功能演示)
- [项目结构](#-项目结构)
- [API接口](#-api-接口)
- [前端应用](#-前端应用)
- [常见问题](#-常见问题-faq)
- [参考资料](#-参考资料)

---

## 🎯 项目简介

**AgentKnowledgeHub** 包含 **4个核心Agent**，通过 [LangGraph](https://langchain-ai.github.io/langgraph/) 有向图编排，实现企业知识的全链路智能处理。

### 4个Agent是什么，分别做什么？

| Agent | 中文名 | 职责 | 类比理解 |
|-------|--------|------|----------|
| `DocParserAgent` | 文档解析Agent | 把PDF/图片/表格等各种格式的文档"读懂"，切割成小段落 | 超强秘书，能看懂任何格式的文件 |
| `KnowledgeExtractAgent` | 知识抽取Agent | 从文本中自动提取人名、公司、关系等结构化信息 | 分析师，把信息整理成知识图谱 |
| `QAAgent` | 问答Agent | 接收用户问题，同时查向量库和知识图谱，生成精准答案 | 专家顾问，综合多源信息回答 |
| `KnowledgeUpdateAgent` | 知识更新Agent | 监听文档变更，只更新变化的部分，保持知识库最新 | 勤快管理员，实时维护知识库 |

### 五大技术亮点

| 亮点 | 说明 | 解决什么问题 |
|------|------|-------------|
| **多模态RAG** | 不只处理文字，还能理解PDF里的图片、表格、流程图 | 传统系统只能处理纯文字 |
| **GraphRAG (知识图谱)** | 用图数据库存储实体关系，支持多跳推理 | 纯向量检索无法处理"关系型"和"多步推理"问题 |
| **CDC增量更新** | 文档变了只更新变化的部分 | 传统方案每次全量重建，1000个文档改5个要30分钟 |
| **记忆系统** | 支持会话记忆、用户画像和个性设置 | 提供个性化对话体验 |
| **Checkpoint 容灾恢复** | MongoDB 持久化每步状态，OOM 崩溃后自动恢复 | 进程崩溃导致全部工作丢失，需从头重跑 |

---

## 🏗 系统架构

### 整体架构图

```
┌──────────────────────────────────────────────────────────┐
│                      用户接口层                            │
│         REST API / Vue Web UI / SDK                       │
└──────────────┬───────────────────────────┬───────────────┘
               │                           │
┌──────────────▼───────────────────────────▼───────────────┐
│            编排引擎 (LangGraph 有向图 + Checkpoint)         │
│    ┌─────────────┬──────────────┬──────────────┐         │
│    │ 文档入库流程  │   问答流程    │  增量更新流程  │         │
│    └──────┬──────┴──────┬───────┴──────┬───────┘         │
└───────────│─────────────│──────────────│─────────────────┘
            │             │              │
┌───────────▼──┐ ┌───────▼────┐ ┌───────▼──────┐ ┌────────────┐
│ 文档解析Agent │ │  问答Agent  │ │ 知识更新Agent │ │ 知识抽取Agent│
│              │ │            │ │              │ │            │
│ - PDF解析    │ │ - 意图识别  │ │ - 文件监听    │ │ - NER实体识别│
│ - 图片OCR    │ │ - 向量检索  │ │ - CDC消费    │ │ - 关系抽取  │
│ - 表格提取   │ │ - 图谱检索  │ │ - 差量对比    │ │ - 事件抽取  │
│ - 文档分块   │ │ - 混合排序  │ │ - 增量更新    │ │ - 三元组生成│
└──────┬───────┘ │ - 答案生成  │ │ - 版本管理    │ └─────┬──────┘
       │         └──┬────┬────┘ └──────┬───────┘       │
       │            │    │             │               │
┌──────▼────────────▼────│─────────────▼───────────────▼──┐
│                        存储层                              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ ChromaDB /  │  │  Neo4j       │  │   SQLite     │     │
│  │ PGVector    │  │  知识图谱     │  │   记忆存储    │     │
│  │ 向量数据库   │  │              │  │              │     │
│  └─────────────┘  └──────────────┘  └──────────────┘     │
│  ┌───────────────────────────────────────────────────┐   │
│  │  MongoDB — LangGraph Checkpoint 状态持久化         │   │
│  │  (每步自动存档，崩溃后可恢复)                        │   │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### 三条工作流水线（每个数据怎么流转的）

**流水线1：文档入库**（上传文档时触发）

```
用户上传文档
     │
     ▼
文档解析Agent  ←── 支持 PDF / Word / Excel / 图片 / Markdown
  ├── 识别文件类型
  ├── 解析内容（文字 + 图片OCR + 表格提取）
  └── 切割成小块（Chunk）
     │
     ▼
知识抽取Agent
  ├── 命名实体识别（NER）：找出人名、公司名、地名等
  ├── 关系抽取：找出实体之间的关系
  └── 生成三元组：("张三", "就职于", "腾讯")
     │
     ├──────────────────────────────┐
     ▼                              ▼
存入向量数据库                   存入知识图谱
(ChromaDB/PGVector)              (Neo4j)
```

**流水线2：智能问答**（用户提问时触发）

```
用户提问："张三负责什么业务？和李四有什么合作关系？"
     │
     ▼
意图识别 + 查询改写
     │
     ├──────────────────┐
     ▼                  ▼
向量检索              图谱检索
(语义相似度)          (关系路径查询)
     │                  │
     └────────┬─────────┘
              ▼
         混合重排序
     (图谱结果权重更高，因为更精准)
              │
              ▼
         LLM生成答案
              │
              ▼
    返回答案 + 来源引用
```

**流水线3：增量更新**（文档修改时触发）

```
文档被修改 / 数据库记录更新
     │
     ▼
CDC事件产生（通过文件监听或Kafka）
     │
     ▼
知识更新Agent
  ├── 差量分析：找出哪些部分变了
  ├── 增量解析：只重新处理变化的内容
  └── 版本管理：记录更新时间和版本号
     │
     ├──────────────┐
     ▼              ▼
更新向量库        更新知识图谱
```

---

## 🛠 技术栈

### 后端技术栈

| 组件 | 技术选型 | 为什么选它 |
|------|----------|------------|
| **Agent编排** | [LangGraph](https://langchain-ai.github.io/langgraph/) | 生产级Agent编排标准，有向图 + Checkpoint 状态持久化 |
| **LLM调用** | [LangChain](https://python.langchain.com/) + OpenAI | 最成熟的LLM应用框架，支持几十种LLM |
| **向量数据库** | [ChromaDB](https://www.trychroma.com/) / [PGVector](https://github.com/pgvector/pgvector) | ChromaDB开箱即用；PGVector适合已有PostgreSQL的企业 |
| **知识图谱** | [Neo4j](https://neo4j.com/) | 图数据库的事实标准，Cypher查询语言强大 |
| **Checkpoint存储** | [MongoDB](https://www.mongodb.com/) + [langgraph-checkpoint-mongodb](https://github.com/langchain-ai/langgraph) | LangGraph 原生支持，自动持久化每步状态，崩溃可恢复 |
| **消息队列** | [Apache Kafka](https://kafka.apache.org/) | CDC事件流处理的工业标准 |
| **API框架** | [FastAPI](https://fastapi.tiangolo.com/) | 异步高性能，自动生成OpenAPI/Swagger文档 |
| **文档解析** | [Unstructured](https://unstructured.io/) + PyPDF2 + Tesseract | 多模态文档解析全家桶 |
| **容器化** | [Docker Compose](https://docs.docker.com/compose/) | 一键启动所有依赖服务 |

### 前端技术栈

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| **框架** | [Vue 3](https://vuejs.org/) | 渐进式JavaScript框架，Composition API |
| **UI组件库** | [Element Plus](https://element-plus.org/) | Vue 3生态最成熟的UI组件库 |
| **HTTP客户端** | [Axios](https://axios-http.com/) | 流行的HTTP客户端库 |

---

## 🚀 快速开始

### 前置条件

在开始之前，你需要安装：

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)（用于一键启动依赖服务）
- Python 3.10+（后端开发）
- Node.js 18+（前端开发）
- 一个 OpenAI API Key（或者用国内兼容接口，见下方FAQ）

### 步骤1：克隆项目

```bash
git clone https://github.com/bcefghj/agent-knowledge-hub.git
cd agent-knowledge-hub
```

### 步骤2：配置环境变量

```bash
cd python
cp .env.example .env
```

### 模型 Provider 配置

Chat 与 Embedding 可以独立选择兼容 OpenAI API 的供应商；推荐使用显式配置，且不要提交 `python/.env`。

```env
# Qwen Chat + Qwen Embedding
CHAT_PROVIDER=qwen
CHAT_API_KEY=your-chat-api-key
CHAT_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
CHAT_MODEL=qwen-plus
EMBEDDING_PROVIDER=qwen
EMBEDDING_API_KEY=your-embedding-api-key
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=1536
```

DeepSeek 目前只用于 Chat，可与 Qwen Embedding 组合：将 `CHAT_PROVIDER` 设为 `deepseek`，并填写 DeepSeek 的 Chat 地址、密钥和模型；保留上面的 `EMBEDDING_*` 配置。不要把 DeepSeek 当作 Embedding Provider。

旧项目配置仍兼容：`CHAT_*` 会回退到 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`；`EMBEDDING_*` 的密钥和地址会回退到相同的 `OPENAI_*`，模型与维度继续使用 `EMBEDDING_MODEL`、`EMBEDDING_DIMENSIONS`。新配置优先。

切换 Embedding 模型或维度可能与已存在的向量 collection 不兼容；生产数据应新建 collection 或按计划重建数据，不能直接混用维度。密钥只应保存在本地 `.env` 或受管密钥服务，绝不写入源码、日志或 `.env.example`。

无网络单元测试：`cd python && python -m pytest tests -q`。真实 smoke test 则在启动服务后调用 `/api/health`、`/api/admin/stats` 和一次简短的 `/api/qa/ask`。

用任意编辑器打开 `.env`，填入你的配置：

```env
# OpenAI配置（必填）
OPENAI_API_KEY=sk-你的APIKey
OPENAI_BASE_URL=https://api.openai.com/v1  # 国内用户可替换为兼容接口地址

# 数据库配置（使用Docker默认值即可，不用改）
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
CHROMA_HOST=localhost
CHROMA_PORT=8000
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# MongoDB（LangGraph Checkpoint，使用Docker默认值即可）
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=agenthub
```

### 步骤3：一键启动所有依赖服务

```bash
# 回到项目根目录
cd ..

# 启动所有依赖（Neo4j、ChromaDB、Kafka、MongoDB）
docker-compose up -d
```

等待约1分钟，然后检查服务状态：

```bash
docker-compose ps
```

你应该看到所有服务状态为 `Up`（共5个服务：neo4j、chromadb、zookeeper、kafka、mongodb）。

### 步骤4：启动Python API服务

```bash
cd python
pip install -r requirements.txt
python -m api.main
```

看到 `Uvicorn running on http://0.0.0.0:8080` 就说明启动成功了！

### 步骤5：启动前端服务（可选）

```bash
cd fontend/vue-aapp
npm install
npm run serve
```

前端服务将在 `http://localhost:8081` 启动。

### 步骤6：验证服务

打开浏览器访问 [http://localhost:8080/docs](http://localhost:8080/docs)，可以看到交互式API文档。

或者用命令行：

```bash
# 健康检查
curl http://localhost:8080/api/health

# 上传一个文档
curl -X POST http://localhost:8080/api/ingest/upload \
  -F "file=@你的文档.pdf"

# 提问
curl -X POST http://localhost:8080/api/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "这个文档讲了什么？"}'
```

---

## 🎬 功能演示

### 功能1：多模态文档解析

文档解析Agent可以自动识别文件类型，调用对应的解析器：

```python
from agents.doc_parser_agent import DocParserAgent

agent = DocParserAgent()

# 解析不同格式的文档
chunks = await agent.parse("年度报告.pdf")    # PDF → 文字 + 图片识别 + 表格提取
chunks = await agent.parse("组织架构.png")    # 图片 → OCR文字识别 + LLM视觉理解
chunks = await agent.parse("财务数据.xlsx")   # Excel → 结构化文本
chunks = await agent.parse("产品文档.md")     # Markdown → 纯文本

# 每个chunk包含：
# chunk.text      - 文本内容
# chunk.metadata  - 来源文件、页码、类型等
# chunk.embedding - 向量表示（自动生成）
```

### 功能2：知识图谱自动构建

知识抽取Agent从文本中提取三元组，自动构建知识图谱：

```python
from agents.knowledge_extract_agent import KnowledgeExtractAgent

extractor = KnowledgeExtractAgent()
result = await extractor.extract(chunks)

# 输出示例：
# entities（实体）:
#   - ("张三", Person, {"职位": "CEO", "年龄": "45"})
#   - ("腾讯", Organization, {"行业": "互联网", "规模": "大型"})
#   - ("微信", Product, {"类型": "社交软件"})
#
# relations（关系）:
#   - ("张三", "就职于", "腾讯")
#   - ("腾讯", "开发了", "微信")
#   - ("张三", "负责", "微信")
```

在Neo4j浏览器（访问 [http://localhost:7474](http://localhost:7474)）中可以可视化查看知识图谱。

### 功能3：GraphRAG 混合检索问答

问答Agent同时从向量库和知识图谱中检索，结合两个来源的信息生成答案：

```python
from agents.qa_agent import QAAgent
from services.vector_store import VectorStoreService
from services.knowledge_graph import KnowledgeGraphService

# 初始化
vs = VectorStoreService()
kg = KnowledgeGraphService()
qa = QAAgent(vector_store=vs, knowledge_graph=kg)

# 提问（支持复杂的多跳推理问题）
result = await qa.answer("张三负责的产品，它的主要竞争对手是谁？")

print(result.answer)     # 生成的自然语言答案
print(result.sources)    # 来源引用（哪些文档/哪些知识图谱节点）
print(result.confidence) # 置信度分数

# 内部执行流程：
# 1. 向量检索 → 找到语义相关的文档段落（用余弦相似度）
# 2. 实体链接 → 识别问题中的"张三"是哪个实体
# 3. 图谱检索 → 张三 → 负责 → 微信 → 竞争对手 → QQ / 钉钉
# 4. 混合重排序 → 图谱路径结果权重×1.25（推理链更精准）
# 5. LLM生成 → 综合所有信息，生成结构化答案
```

### 功能4：CDC 增量更新（只更新变化的部分）

```python
from agents.knowledge_update_agent import KnowledgeUpdateAgent

update_agent = KnowledgeUpdateAgent(...)

# 场景：你修改了一个PDF文件的第3页

# ❌ 传统做法（全量更新）：
#   1. 删除该文档所有向量 （删 1000 条）
#   2. 重新解析整个PDF     （解析 50 页）
#   3. 重新入库所有内容    （写入 1000 条）
#   耗时：~30 分钟

# ✅ CDC做法（增量更新）：
#   1. 检测到第3页内容变化
#   2. 只重新解析第3页
#   3. 只更新第3页相关的向量和知识图谱节点
#   耗时：~30 秒（快60倍！）

await update_agent.process_cdc_event(event={
    "operation": "UPDATE",
    "resource_path": "/docs/年度报告.pdf",
    "changed_pages": [3]
})
```

### 功能5：记忆系统

支持会话管理、用户画像和AI个性设置：

```python
from services.memory_service import MemoryService

memory = MemoryService()

# 创建对话会话
conversation = await memory.create_conversation(
    session_id="session_001",
    user_id="user_001",
    title="年度报告咨询"
)

# 保存对话消息
await memory.save_message(
    session_id="session_001",
    role="user",
    content="张三的职位是什么？"
)

# 获取用户画像
profile = await memory.get_user_profile("user_001")

# 设置AI个性参数
await memory.update_personality("user_001", warmth=80, expertise=90, humor=60, empathy=75)
```

### 功能6：Checkpoint 容灾恢复（OOM 崩溃不怕丢数据）

每个工作流节点执行完后，状态自动持久化到 MongoDB。进程崩溃后用相同 `thread_id` 即可从最近检查点恢复：

```python
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from orchestrator.graph import build_knowledge_graph_workflow

# 创建 MongoDB checkpointer
mongo_client = MongoClient("mongodb://localhost:27017")
checkpointer = MongoDBSaver(client=mongo_client, db_name="agenthub")

# 构建带 checkpointer 的工作流
workflows = build_knowledge_graph_workflow(
    vector_store=vs,
    knowledge_graph=kg,
    memory_service=memory,
    checkpointer=checkpointer,
)

# 第一次调用 — 正常执行
config = {"configurable": {"thread_id": "ingest-report.pdf"}}
result = await workflows["ingest"].ainvoke(
    {"file_paths": ["report.pdf"], "request_id": "req-001"},
    config=config,
)
# 节点执行顺序: parse → extract → store_vectors → store_graph
# 每步自动写入 MongoDB checkpoint

# ⚠️ 假设进程在 store_vectors 节点 OOM 崩溃了...

# 重启后用相同 thread_id 调用 — 自动从 checkpoint 恢复
result = await workflows["ingest"].ainvoke(
    {"file_paths": ["report.pdf"], "request_id": "req-001"},
    config=config,
)
# LangGraph 检测到 parse、extract 已有 checkpoint → 跳过
# store_graph 重新执行（幂等保护确保不重复写入）
```

**核心机制：**

| 特性 | 说明 |
|------|------|
| **自动 Checkpoint** | 每个节点执行完后，`MongoDBSaver` 自动将状态写入 MongoDB |
| **thread_id 恢复** | 用相同 `thread_id` 调用 `ainvoke()`，LangGraph 自动加载最近 checkpoint |
| **幂等保护** | `IdempotentNode` 装饰器通过 `request_id` 去重，防止重试产生副作用 |
| **超时重试** | 节点执行超时后自动重试，指数退避（2s → 4s → 8s），最多 3 次 |
| **死循环防护** | Update Pipeline 最多重试 3 次，`retry_count` 累加控制 |

---

## 📁 项目结构

```
AgentKnowledgeHub/
│
├── README.md                          ← 你正在看的这个文件
├── docker-compose.yml                 ← 一键启动所有依赖服务（Neo4j + ChromaDB + Kafka + MongoDB）
├── uploads/                           ← 默认文件上传目录
│
├── doc/                               ← 项目文档资料
│
├── docs/                              ← 技术文档目录
│   ├── architecture.md                ← 架构设计详解（每个决策的理由）
│   ├── project-plan.md               ← 项目规划方案
│   └── tech-deep-dive.md              ← 核心代码逐行讲解
│
├── python/                            ← Python后端实现（功能最完整）
│   ├── agents/                        ← 4个核心Agent
│   │   ├── doc_parser_agent.py        ← 文档解析Agent
│   │   ├── knowledge_extract_agent.py ← 知识抽取Agent
│   │   ├── qa_agent.py                ← 问答Agent
│   │   └── knowledge_update_agent.py  ← 知识更新Agent
│   ├── orchestrator/
│   │   └── graph.py                   ← LangGraph编排引擎（3条流水线 + MongoDB Checkpoint + 幂等重试）
│   ├── services/
│   │   ├── vector_store.py            ← 向量库服务（ChromaDB/PGVector）
│   │   ├── knowledge_graph.py         ← 知识图谱服务（Neo4j）
│   │   ├── graph_rag.py               ← GraphRAG混合检索管道
│   │   ├── cdc_processor.py           ← CDC增量更新处理器
│   │   ├── multimodal.py              ← 多模态处理服务
│   │   ├── memory_service.py          ← 记忆服务
│   │   └── memory_models.py           ← 记忆数据模型
│   ├── api/
│   │   └── main.py                    ← FastAPI入口（REST API）
│   ├── config/
│   │   └── settings.py                ← 配置管理
│   ├── tests/                         ← 测试目录
│   ├── uploads/                       ← 文档上传目录
│   ├── Dockerfile                     ← Python服务容器化
│   ├── requirements.txt               ← Python依赖
│   ├── .env                           ← 环境变量配置
│   └── .env.example                   ← 环境变量模板
│
└── fontend/                           ← 前端应用
    └── vue-aapp/                      ← Vue 3 + Element Plus应用
        ├── public/                    ← 静态资源
        ├── src/
        │   ├── api/                   ← API调用封装
        │   ├── assets/                ← 静态资源
        │   ├── components/            ← Vue组件
        │   │   ├── ApiDemo.vue        ← API演示组件
        │   │   ├── ConversationManager.vue  ← 对话管理
        │   │   ├── ConversationWorkbench.vue ← 对话工作台
        │   │   ├── KnowledgeBase.vue  ← 知识库管理
        │   │   ├── LoginPage.vue      ← 登录页面
        │   │   ├── PersonalitySettings.vue ← 个性设置
        │   │   └── ProfileSettings.vue ← 用户画像设置
        │   ├── composables/           ← Vue组合式函数
        │   ├── App.vue                ← 根组件
        │   └── main.js                ← 入口文件
        └── package.json               ← 前端依赖
```

---

## 📡 API 接口

启动服务后，访问 [http://localhost:8080/docs](http://localhost:8080/docs) 查看交互式 Swagger API 文档。

### 文档管理接口

| 方法 | 路径 | 说明 | 示例 |
|------|------|------|------|
| `POST` | `/api/ingest/upload` | 上传单个文档 | `curl -F "file=@doc.pdf" http://localhost:8080/api/ingest/upload` |
| `POST` | `/api/ingest/batch` | 批量上传文档 | 上传多个文件，自动并行处理 |
| `GET` | `/api/ingest/documents` | 获取已上传文档列表 | - |
| `DELETE` | `/api/ingest/documents/{file_name}` | 删除指定文档 | - |

### 智能问答接口

| 方法 | 路径 | 说明 | 请求体示例 |
|------|------|------|-----------|
| `POST` | `/api/qa/ask` | 智能问答 | `{"question": "张三的职位？", "session_id": "xxx", "user_id": "xxx"}` |

**响应示例：**
```json
{
  "question": "张三的职位？",
  "answer": "根据文档，张三担任腾讯公司CEO职务，负责微信产品线。",
  "confidence": 0.94,
  "intent": "qa",
  "sources": [
    {"content": "文档内容摘要...", "source": "年度报告.pdf", "score": 0.92, "type": "vector"},
    {"content": "张三-就职于-腾讯", "source": "knowledge_graph", "score": 0.98, "type": "graph"}
  ],
  "reasoning_steps": ["步骤1: 实体识别", "步骤2: 向量检索", "步骤3: 图谱推理"]
}
```

### 记忆管理接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/memory/profile/{user_id}` | 获取用户画像 |
| `POST` | `/api/memory/profile` | 更新用户画像 |
| `GET` | `/api/memory/personality/{user_id}` | 获取AI个性参数 |
| `POST` | `/api/memory/personality` | 更新AI个性参数 |
| `POST` | `/api/memory/retrieve` | 检索相关历史记忆 |

### 对话管理接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/conversations` | 创建新对话会话 |
| `GET` | `/api/conversations` | 获取对话列表 |
| `POST` | `/api/conversations/search` | 搜索对话 |
| `GET` | `/api/conversations/{session_id}` | 获取会话详情 |
| `POST` | `/api/conversations/{session_id}/messages` | 添加消息 |
| `PUT` | `/api/conversations/{session_id}/title` | 更新会话标题 |
| `DELETE` | `/api/conversations/{session_id}` | 删除会话 |

### 管理接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/admin/stats` | 查看系统统计（文档数、实体数、关系数） |
| `POST` | `/api/admin/update` | 手动触发增量更新 |
| `GET` | `/api/health` | 健康检查 |

---

## 🖥 前端应用

项目包含一个完整的 Vue 3 前端应用，位于 `fontend/vue-aapp/` 目录。

### 前端功能

| 功能模块 | 说明 |
|----------|------|
| **登录页面** | 用户认证入口 |
| **对话工作台** | 智能问答界面，支持上下文保持 |
| **知识库管理** | 文档上传、列表管理、删除操作 |
| **个性设置** | 调整AI助手的温暖度、专业度、幽默感、共情力 |
| **用户画像** | 管理用户个人信息和偏好设置 |

### 启动前端

```bash
cd fontend/vue-aapp
npm install
npm run serve
```

访问 [http://localhost:8081](http://localhost:8081) 即可使用前端界面。

---

## ❓ 常见问题 FAQ

### Q: 我没有OpenAI API Key怎么办？

完全没问题！可以用任何兼容OpenAI接口的LLM服务：

```env
# 国内免费/便宜的选择：
# 1. 通义千问（阿里）
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=你的通义千问APIKey

# 2. 智谱AI（GLM系列）
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
OPENAI_API_KEY=你的智谱APIKey

# 3. 本地部署（完全免费）
# 先安装 Ollama: https://ollama.ai/
# 然后 ollama pull qwen2
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen2
```

### Q: Docker启动后服务报错？

```bash
# 查看所有服务状态
docker-compose ps

# 查看某个服务的日志
docker-compose logs neo4j
docker-compose logs kafka
docker-compose logs mongodb

# 重启某个服务
docker-compose restart neo4j
```

Neo4j需要的内存比较多，建议给Docker分配至少4GB内存（Docker Desktop → Settings → Resources → Memory）。

### Q: 如何验证 MongoDB Checkpoint 是否正常工作？

```bash
# 查看 MongoDB 中的 checkpoint 数据
docker exec agenthub-mongodb mongosh --eval \
  "db.getSiblingDB('agenthub').checkpoints.find().toArray()"

# 如果有数据说明 checkpoint 正常写入
# 每次 ainvoke() 调用后，checkpoint 集合会自动新增记录
```

### Q: 这个项目可以直接用在公司生产环境吗？

这是一个**架构展示 + 学习项目**，展示了企业级系统的设计思路。如果要用在生产环境，还需要补充：

- 用户认证和权限控制（JWT / OAuth2）
- API限流和熔断（防止滥用）
- 完善的日志和监控（ELK Stack / Prometheus）
- 全面的单元测试和集成测试
- 生产级的数据备份方案

### Q: 如何运行测试？

```bash
# Python
cd python
pytest tests/
```

### Q: 前端启动失败？

```bash
# 确保Node.js版本 >= 18
node --version

# 清除缓存重新安装
rm -rf node_modules package-lock.json
npm install

# 检查端口是否被占用
lsof -i :8081
```

---

## 🔗 参考资料

### 核心框架文档

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/) — Agent编排框架
- [LangChain 官方文档](https://python.langchain.com/docs/get_started/introduction) — LLM应用框架
- [FastAPI 官方文档](https://fastapi.tiangolo.com/) — Python API框架
- [Neo4j 官方文档](https://neo4j.com/docs/) — 图数据库
- [ChromaDB 官方文档](https://docs.trychroma.com/) — 向量数据库
- [Vue 3 官方文档](https://vuejs.org/guide/) — 前端框架

### 关键论文

- [RAG原始论文 (2020)](https://arxiv.org/abs/2005.11401) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- [GraphRAG论文 (2024)](https://arxiv.org/abs/2404.16130) — From Local to Global: A Graph RAG Approach to Query-Focused Summarization
- [Microsoft GraphRAG 开源项目](https://github.com/microsoft/graphrag)

### 相关学习资源

- [LangGraph教程（官方）](https://langchain-ai.github.io/langgraph/tutorials/)
- [Neo4j Graph Academy](https://graphacademy.neo4j.com/) — 免费图数据库课程
- [Vue 3 Composition API 教程](https://vuejs.org/guide/extras/composition-api-faq.html)

---

## 🤝 贡献

欢迎提 Issue 和 PR！

- 发现 Bug？[提交 Issue](https://github.com/bcefghj/agent-knowledge-hub/issues)
- 想加新功能？欢迎 Fork 后提 PR
- 觉得有帮助？请点个 ⭐ Star，这是对我最大的鼓励！

## 📄 License

[MIT License](./LICENSE) — 可以自由使用、修改、分发，只需保留原始版权声明。
