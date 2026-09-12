"""
FastAPI 入口 — 企业知识管理系统 REST API

提供三组接口:
  1. /api/ingest   — 文档上传 & 入库
  2. /api/qa       — 智能问答
  3. /api/admin    — 管理（统计、更新触发）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Body, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from orchestrator.graph import build_knowledge_graph_workflow
from services.knowledge_graph import KnowledgeGraphService
from services.memory_service import MemoryService
from services.vector_store import VectorStoreService
from services.document_registry import DocumentRegistry, RegistryError, safe_error
from services.document_processor import (
    DocumentParseError,
    DocumentProcessorAdapter,
    EmptyDocumentError,
    InvalidExtractionResult,
    KnowledgeExtractionError,
    ProcessingTimeoutError,
    UnsupportedDocumentType,
    safe_filename,
)
from services.document_update_coordinator import (
    DocumentUpdateCoordinator,
    OperationBusyError,
    OperationIdentityConflictError,
)
from providers.factory import create_chat_provider, create_embedding_provider
from agents.doc_parser_agent import DocParserAgent
from agents.knowledge_extract_agent import KnowledgeExtractAgent

knowledge_graph = KnowledgeGraphService()
vector_store: VectorStoreService | None = None
memory_service: MemoryService | None = None
workflows: dict[str, Any] = {}
mongo_client = None
document_registry: DocumentRegistry | None = None
document_coordinator: DocumentUpdateCoordinator | None = None
MAX_DOCUMENT_UPLOAD_BYTES = 25 * 1024 * 1024


def build_document_coordinator(
    registry: DocumentRegistry,
    vectors: VectorStoreService,
    graph: KnowledgeGraphService,
    chat_provider: Any,
    *,
    temp_root: str | Path,
) -> DocumentUpdateCoordinator:
    """Compose document dependencies from existing lifecycle-owned clients only."""
    processor = DocumentProcessorAdapter(
        DocParserAgent(chat_provider), KnowledgeExtractAgent(chat_provider), temp_root=temp_root
    )
    return DocumentUpdateCoordinator(registry, vectors, graph, processor)

# 初始化知识图谱和工作流
@asynccontextmanager
async def lifespan(app: FastAPI):
    """初始化知识图谱和工作流"""
    global mongo_client, vector_store, memory_service, document_registry, document_coordinator
    os.makedirs(settings.upload_dir, exist_ok=True)   # 确保上传目录存在
    chat_provider = create_chat_provider(settings)
    embedding_provider = create_embedding_provider(settings)
    vector_store = VectorStoreService(embedding_provider)
    memory_service = MemoryService(embedding_provider)
    await vector_store.init()    # 初始化向量存储；失败时阻止服务伪健康启动
    await knowledge_graph.init()    # 初始化知识图谱；失败时阻止服务伪健康启动

    # 初始化 MongoDB checkpointer
    from pymongo import MongoClient
    from langgraph.checkpoint.mongodb import MongoDBSaver
    mongo_client = MongoClient(settings.mongodb_uri)
    document_registry = DocumentRegistry(mongo_client[settings.mongodb_database])
    document_registry.ensure_indexes()
    document_coordinator = build_document_coordinator(
        document_registry,
        vector_store,
        knowledge_graph,
        chat_provider,
        temp_root=Path(settings.upload_dir) / ".processing",
    )
    app.state.document_registry = document_registry
    app.state.document_coordinator = document_coordinator
    checkpointer = MongoDBSaver(
        client=mongo_client,
        db_name=settings.mongodb_database,
    )

    workflows.update(                   # 初始化知识图谱工作流
        build_knowledge_graph_workflow(
            chat_provider=chat_provider,
            vector_store=vector_store,
            knowledge_graph=knowledge_graph,
            memory_service=memory_service,
            checkpointer=checkpointer,
        )
    )
    yield
    await knowledge_graph.close()
    if mongo_client:
        mongo_client.close()
    document_coordinator = None
    app.state.document_coordinator = None


app = FastAPI(         # 初始化FastAPI应用
    title="AgentKnowledgeHub — 多Agent企业知识管理系统",
    description="支持多模态RAG、知识图谱、增量更新的企业级知识管理 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(    # 添加CORS中间件，允许跨域请求
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ────────────────────────────────

class QuestionRequest(BaseModel):  # 问答请求模型
    """问答请求模型"""
    question: str
    session_id: str | None = None
    user_id: str | None = None


class QuestionResponse(BaseModel):  # 问答响应模型
    """问答响应模型"""
    question: str
    answer: str
    confidence: float
    intent: str
    sources: list[dict[str, Any]]
    reasoning_steps: list[str]


class IngestResponse(BaseModel):  # 文档入库响应模型
    """文档入库响应模型"""
    file_name: str
    chunks_count: int
    entities_count: int
    relations_count: int
    status: str
    document_id: str | None = None
    namespace: str = "default"
    logical_key: str | None = None
    version: int | None = None
    content_hash: str | None = None
    changed: bool = True
    operation_id: str | None = None
    status_url: str | None = None
    completed_steps: list[str] = []


class StatsResponse(BaseModel):  # 统计响应模型
    """统计响应模型"""
    vector_store: dict[str, Any]
    knowledge_graph: dict[str, Any]


# ── Versioned document endpoints ─────────────────────────────

def _registry() -> DocumentRegistry:
    registry = getattr(app.state, "document_registry", None) or document_registry
    if registry is None:
        raise HTTPException(status_code=503, detail="Document registry is unavailable")
    return registry


def _coordinator() -> DocumentUpdateCoordinator:
    coordinator = getattr(app.state, "document_coordinator", None) or document_coordinator
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Document processing is unavailable")
    return coordinator


async def _read_document_upload(file: UploadFile) -> tuple[str, bytes]:
    file_name = safe_filename(file.filename or "")
    suffix = Path(file_name).suffix.lower()
    if not file_name or suffix not in DocParserAgent.SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported document type")
    content = await file.read(MAX_DOCUMENT_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Document is empty")
    if len(content) > MAX_DOCUMENT_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Document exceeds the upload size limit")
    return file_name, content


def _safe_operation_response(result: dict[str, Any], file_name: str, *, namespace: str = "default", logical_key: str | None = None) -> IngestResponse:
    operation_id = result.get("operation_id")
    metadata = result.get("processing_metadata", {})
    return IngestResponse(
        file_name=file_name,
        chunks_count=int(metadata.get("chunk_count", 0)),
        entities_count=int(metadata.get("entity_count", 0)),
        relations_count=int(metadata.get("relation_count", 0)),
        status=str(result.get("status", "processing")),
        document_id=result.get("document_id"),
        namespace=namespace,
        logical_key=logical_key,
        version=result.get("version"),
        content_hash=result.get("content_hash"),
        changed=bool(result.get("changed", True)),
        operation_id=operation_id,
        status_url=f"/api/document-operations/{operation_id}" if operation_id else None,
        completed_steps=list(result.get("completed_steps", [])),
    )


def _raise_document_error(error: Exception) -> None:
    if isinstance(error, HTTPException):
        raise error
    if isinstance(error, (UnsupportedDocumentType, EmptyDocumentError)):
        raise HTTPException(status_code=400, detail="Invalid document input") from None
    if isinstance(error, (InvalidExtractionResult, DocumentParseError, KnowledgeExtractionError)):
        raise HTTPException(status_code=422, detail="Document processing produced an invalid result") from None
    if isinstance(error, ProcessingTimeoutError):
        raise HTTPException(status_code=504, detail="Document processing timed out") from None
    if isinstance(error, OperationBusyError):
        raise HTTPException(status_code=409, detail="A document operation is already active") from None
    if isinstance(error, OperationIdentityConflictError):
        raise HTTPException(status_code=409, detail="Operation ID is already bound to another request") from None
    if isinstance(error, RegistryError):
        message = str(error).lower()
        if "not found" in message:
            raise HTTPException(status_code=404, detail="Document not found") from None
        if "invalid" in message or "processing" in message or "deleted" in message:
            raise HTTPException(status_code=409, detail="Document state conflicts with this operation") from None
        raise HTTPException(status_code=503, detail="Document registry is unavailable") from None
    raise HTTPException(status_code=503, detail="A required document-processing dependency is unavailable") from None


def _optional_operation_id(value: str | None) -> str:
    if value is None:
        return str(uuid.uuid4())
    try:
        uuid.UUID(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="X-Operation-Id must be a UUID") from None
    return value


async def _create_document(
    file: UploadFile,
    logical_key: str | None,
    namespace: str,
    operation_id: str | None = None,
) -> IngestResponse:
    file_name, content = await _read_document_upload(file)
    operation_id = _optional_operation_id(operation_id)
    try:
        result = await _coordinator().create_document_version(
            filename=file_name, content=content, logical_key=logical_key, namespace=namespace, operation_id=operation_id
        )
    except Exception as exc:
        _raise_document_error(exc)
    return _safe_operation_response(result, file_name, namespace=namespace, logical_key=logical_key)


@app.post("/api/documents", response_model=IngestResponse, tags=["文档入库"])
@app.post("/api/ingest/upload", response_model=IngestResponse, tags=["文档入库"])
async def upload_document(
    file: UploadFile = File(...),
    logical_key: str | None = Form(None),
    namespace: str = Form("default"),
    operation_id: str | None = Header(None, alias="X-Operation-Id"),
):
    """Create a versioned document; the legacy URL remains a compatibility alias."""
    return await _create_document(file, logical_key, namespace, operation_id)

@app.get("/api/documents/{document_id}")
async def get_registered_document(document_id: str):
    row = _registry().find(document_id)
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return row

@app.get("/api/documents/{document_id}/versions")
async def get_document_versions(document_id: str):
    if not _registry().find(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return _registry().versions_for(document_id)

@app.get("/api/documents/{document_id}/status")
async def get_document_status(document_id: str):
    return await get_registered_document(document_id)


@app.put("/api/documents/{document_id}", response_model=IngestResponse, tags=["文档入库"])
async def update_registered_document(
    document_id: str, file: UploadFile = File(...)
):
    """Create the next immutable version for the path document ID only."""
    file_name, content = await _read_document_upload(file)
    document = _registry().find(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if document.get("status") in {"processing", "deleting"}:
        raise HTTPException(status_code=409, detail="Document is currently being processed")
    operation_id = str(uuid.uuid4())
    try:
        result = await _coordinator().update_document(
            document_id, filename=file_name, content=content, operation_id=operation_id
        )
    except Exception as exc:
        _raise_document_error(exc)
    return _safe_operation_response(
        result, file_name, namespace=document.get("namespace", "default"), logical_key=document.get("logical_key")
    )


@app.delete("/api/documents/{document_id}", tags=["文档入库"])
async def delete_registered_document(document_id: str):
    """Run the document-scoped delete Saga; never accepts a path or source."""
    document = _registry().find(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if document.get("status") in {"processing", "deleting"}:
        raise HTTPException(status_code=409, detail="Document is currently being processed")
    if document.get("status") == "deleted":
        return {"document_id": document_id, "operation_id": None, "status": "deleted", "changed": False}
    operation_id = str(uuid.uuid4())
    try:
        result = await _coordinator().delete_document(document_id, operation_id=operation_id)
    except Exception as exc:
        _raise_document_error(exc)
    return {
        "document_id": document_id,
        "operation_id": result.get("operation_id", operation_id),
        "status": result.get("status", "processing"),
        "changed": True,
        "status_url": f"/api/document-operations/{result.get('operation_id', operation_id)}",
    }


@app.get("/api/document-operations/{operation_id}", tags=["文档入库"])
async def get_document_operation(operation_id: str):
    operation = _coordinator().journal.get(operation_id)
    if not operation:
        raise HTTPException(status_code=404, detail="Document operation not found")
    payload = {
        key: operation.get(key)
        for key in (
            "operation_id", "operation_type", "document_id", "version", "status",
            "completed_steps", "compensation_steps", "created_at", "updated_at",
            "error_phase", "error_category", "error_type", "chunk_index",
        )
    } | {"error_summary": safe_error(operation.get("error_summary") or "") or None}
    document_id = operation.get("document_id")
    if document_id:
        document = _registry().find(document_id)
        payload["document_status"] = document.get("status") if document else None
    return payload


@app.post("/api/ingest/batch", response_model=list[IngestResponse], tags=["文档入库"])
async def upload_batch(files: list[UploadFile] = File(...)):
    """批量上传文档"""
    results = []
    for file in files:
        resp = await _create_document(file, logical_key=None, namespace="default")
        results.append(resp)
    return results


@app.get("/api/ingest/documents", tags=["文档入库"])
async def get_documents():
    """Legacy list URL with its established fields plus registry identity/state."""
    if document_registry or getattr(app.state, "document_registry", None):
        rows = []
        for document in _registry().list_documents():
            versions = _registry().versions_for(document["document_id"])
            current = next((item for item in versions if item.get("is_current")), {})
            updated_at = document.get("updated_at")
            rows.append({
                "id": document["document_id"], "name": document.get("filename", "unnamed"),
                "size": 0, "upload_time": updated_at.timestamp() if hasattr(updated_at, "timestamp") else 0,
                "chunks_count": current.get("chunk_count", 0), "document_id": document["document_id"],
                "version": document.get("current_version"), "status": document.get("status"),
                "changed": True, "legacy": False,
            })
        known_names = {row["name"] for row in rows}
        if os.path.exists(settings.upload_dir):
            for filename in os.listdir(settings.upload_dir):
                filepath = os.path.join(settings.upload_dir, filename)
                if os.path.isfile(filepath) and filename not in known_names:
                    stat = os.stat(filepath)
                    rows.append({
                        "id": filename, "name": filename, "size": stat.st_size,
                        "upload_time": stat.st_ctime, "chunks_count": 0,
                        "document_id": None, "version": None, "status": "legacy",
                        "changed": False, "legacy": True,
                    })
        return rows
    """Fallback only for an API instance without the registry dependency."""
    documents = []
    if os.path.exists(settings.upload_dir):
        for filename in os.listdir(settings.upload_dir):
            filepath = os.path.join(settings.upload_dir, filename)
            if os.path.isfile(filepath):
                stat = os.stat(filepath)
                chunks_count = 0
                documents.append({
                    "id": filename,
                    "name": filename,
                    "size": stat.st_size,
                    "upload_time": stat.st_ctime,
                    "chunks_count": chunks_count
                })
    return documents


@app.delete("/api/ingest/documents/{file_name}", tags=["文档入库"])
async def delete_document(file_name: str):
    """Reject filename-based deletes; versioned deletion needs a document UUID."""
    safe_name = Path(file_name).name
    if safe_name != file_name:
        raise HTTPException(status_code=400, detail="Invalid file name")
    raise HTTPException(
        status_code=410,
        detail="Filename-based deletion is retired; use DELETE /api/documents/{document_id}.",
    )


# ── QA Endpoints ─────────────────────────────────────────────

@app.post("/api/qa/ask", response_model=QuestionResponse, tags=["智能问答"])
async def ask_question(req: QuestionRequest):
    """智能问答 — 混合检索 + 知识图谱推理 + 记忆系统"""
    if not settings.has_usable_llm_key:
        raise HTTPException(
            status_code=503,
            detail="Question answering requires a configured non-placeholder OPENAI_API_KEY.",
        )
    qa_wf = workflows.get("qa")      ### 从工作流字典，获取智能问答工作流
    if not qa_wf:
        raise HTTPException(status_code=503, detail="QA workflow not initialized")

    thread_id = req.session_id or str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    inputs = {
        "question": req.question,
        "request_id": request_id,
    }     # 初始化输入参数，包含问题和幂等 ID
    if req.session_id:
        inputs["session_id"] = req.session_id    # 如果提供了会话 ID，添加到输入参数
    if req.user_id:
        inputs["user_id"] = req.user_id    # 如果提供了用户 ID，添加到输入参数

    result = await qa_wf.ainvoke(inputs, config=config)
    qa_result = result.get("result")     # 从工作流结果中提取问答结果
    if not qa_result:
        raise HTTPException(status_code=500, detail="QA failed")

    return QuestionResponse(        # 返回问答结果
        question=qa_result.question,     # 问答结果中的问题
        answer=qa_result.answer,       # 问答结果中的答案 
        confidence=qa_result.confidence,  # 问答结果中的置信度
        intent=qa_result.intent.value,    # 问答结果中的意图
        sources=[
            {
                "content": c.content[:200],
                "source": Path(c.source).name,
                "score": c.score,
                "type": c.retrieval_type,
                "document_id": c.metadata.get("document_id"),
                "document_version": c.metadata.get("document_version"),
            }
            for c in qa_result.contexts
        ],
        reasoning_steps=qa_result.reasoning_steps,
    )


# ── Admin Endpoints ──────────────────────────────────────────

@app.get("/api/admin/stats", response_model=StatsResponse, tags=["系统管理"])
async def get_stats():
    """获取系统统计信息"""
    vs_stats = await vector_store.get_stats()
    kg_stats = await knowledge_graph.get_stats()
    return StatsResponse(vector_store=vs_stats, knowledge_graph=kg_stats)


@app.post("/api/admin/update", tags=["系统管理"])
async def trigger_update(payload: dict[str, Any] | None = Body(default=None)):
    """Retire the local-path admin update route without inspecting its payload."""
    del payload
    raise HTTPException(
        status_code=410,
        detail="Legacy admin updates are retired; use POST or PUT /api/documents with multipart bytes.",
    )


# ── Memory Endpoints ─────────────────────────────────────────

class UserProfileRequest(BaseModel):  # 用户配置请求模型
    """用户配置请求模型"""
    user_id: str
    name: str | None = None
    background: str | None = None
    preferences: dict | None = None

class UserProfileResponse(BaseModel):  # 用户配置响应模型
    """用户配置响应模型"""
    user_id: str
    name: str
    background: str
    preferences: dict
    interaction_count: int
    last_active: str
    created_at: str

class PersonalityRequest(BaseModel):  # 人格配置请求模型
    """人格配置请求模型"""
    user_id: str
    warmth: float | None = None
    expertise: float | None = None
    humor: float | None = None
    empathy: float | None = None

class PersonalityResponse(BaseModel):  # 人格配置响应模型
    """人格配置响应模型"""
    user_id: str
    warmth: float
    expertise: float
    humor: float
    empathy: float

class MemoryRetrieveRequest(BaseModel):  # 记忆检索请求模型
    """记忆检索请求模型"""
    query: str
    user_id: str = "default_user"
    top_k: int = 5

class MemoryEventResponse(BaseModel):  # 记忆事件响应模型
    """记忆事件响应模型"""
    session_id: str
    timestamp: str
    user_input: str
    agent_response: str
    summary: str
    topics: list[str]
    importance: float

# ── Conversation Models ───────────────────────────────────────

class MessageRequest(BaseModel):  # 消息请求模型
    """消息请求模型"""
    role: str  # user | assistant | system
    content: str
    message_id: str | None = None
    metadata: dict | None = None

class MessageResponse(BaseModel):  # 消息响应模型
    """消息响应模型"""
    id: str
    session_id: str
    role: str
    content: str
    timestamp: str
    metadata: dict

class ConversationRequest(BaseModel):  # 对话请求模型
    """对话请求模型"""
    session_id: str
    user_id: str | None = None
    title: str | None = None
    metadata: dict | None = None

class ConversationResponse(BaseModel):  # 对话响应模型
    """对话响应模型"""
    session_id: str
    user_id: str
    title: str
    messages: list[MessageResponse]
    created_at: str
    updated_at: str
    metadata: dict

class ConversationListResponse(BaseModel):  # 对话列表响应模型
    """对话列表响应模型"""
    conversations: list[ConversationResponse]
    total: int

class ConversationSearchRequest(BaseModel):  # 对话搜索请求模型
    """对话搜索请求模型"""
    query: str
    user_id: str | None = None
    limit: int = 20

class DeleteMessageRequest(BaseModel):  # 删除消息请求模型
    """删除消息请求模型"""
    message_id: str

@app.get("/api/memory/profile/{user_id}", response_model=UserProfileResponse, tags=["记忆管理"])   # 获取用户画像接口
async def get_user_profile(user_id: str):
    """获取用户画像"""
    profile = await memory_service.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")
    return UserProfileResponse(
        user_id=profile.user_id,
        name=profile.name,
        background=profile.background,
        preferences=profile.preferences,
        interaction_count=profile.interaction_count,
        last_active=profile.last_active,
        created_at=profile.created_at,
    )

@app.post("/api/memory/profile", response_model=UserProfileResponse, tags=["记忆管理"])   # 更新用户画像接口
async def update_user_profile(req: UserProfileRequest):
    """更新用户画像"""
    profile = await memory_service.get_user_profile(req.user_id)
    if not profile:
        from services.memory_models import UserProfile
        profile = UserProfile(user_id=req.user_id)
    
    if req.name is not None:
        profile.name = req.name
    if req.background is not None:
        profile.background = req.background
    if req.preferences is not None:
        profile.preferences.update(req.preferences)
    
    await memory_service.update_user_profile(req.user_id, profile)   # 更新用户画像
    return UserProfileResponse(
        user_id=profile.user_id,
        name=profile.name,
        background=profile.background,
        preferences=profile.preferences,
        interaction_count=profile.interaction_count,
        last_active=profile.last_active,
        created_at=profile.created_at,
    )

@app.get("/api/memory/personality/{user_id}", response_model=PersonalityResponse, tags=["记忆管理"])   # 获取AI个性参数接口
async def get_personality(user_id: str):
    """获取AI个性参数"""
    personality = await memory_service.get_personality(user_id)
    return PersonalityResponse(
        user_id=user_id,
        warmth=personality.warmth,
        expertise=personality.expertise,
        humor=personality.humor,
        empathy=personality.empathy,
    )

@app.post("/api/memory/personality", response_model=PersonalityResponse, tags=["记忆管理"])   # 更新AI个性参数接口
async def update_personality(req: PersonalityRequest):
    """更新AI个性参数"""
    personality = await memory_service.get_personality(req.user_id)
    
    if req.warmth is not None:
        personality.warmth = max(0, min(100, req.warmth))
    if req.expertise is not None:
        personality.expertise = max(0, min(100, req.expertise))
    if req.humor is not None:
        personality.humor = max(0, min(100, req.humor))
    if req.empathy is not None:
        personality.empathy = max(0, min(100, req.empathy))
    
    await memory_service.update_personality(req.user_id, personality)
    return PersonalityResponse(
        user_id=req.user_id,
        warmth=personality.warmth,
        expertise=personality.expertise,
        humor=personality.humor,
        empathy=personality.empathy,
    )

@app.post("/api/memory/retrieve", response_model=list[MemoryEventResponse], tags=["记忆管理"])   # 检索相关历史记忆接口
async def retrieve_memory(req: MemoryRetrieveRequest):
    """检索相关历史记忆"""
    events = await memory_service.retrieve_long_term(req.query, req.top_k)
    return [MemoryEventResponse(
        session_id=e.session_id,
        timestamp=e.timestamp,
        user_input=e.user_input,
        agent_response=e.agent_response,
        summary=e.summary,
        topics=e.topics,
        importance=e.importance,
    ) for e in events]

# ── Conversation Endpoints ────────────────────────────────────

@app.post("/api/conversations", response_model=ConversationResponse, tags=["对话管理"])   # 创建新对话会话接口 
async def create_conversation(req: ConversationRequest):
    """创建新对话会话"""
    conversation = await memory_service.create_conversation(
        session_id=req.session_id,
        user_id=req.user_id or "",
        title=req.title or "",
        metadata=req.metadata
    )
    return ConversationResponse(
        session_id=conversation.session_id,
        user_id=conversation.user_id,
        title=conversation.title,
        messages=[],
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        metadata=conversation.metadata
    )

@app.get("/api/conversations", response_model=list[ConversationResponse], tags=["对话管理"])   # 获取所有历史对话列表接口
async def list_conversations(user_id: str = "", limit: int = 20, offset: int = 0):
    """获取所有历史对话列表"""
    conversations = await memory_service.list_conversations(user_id=user_id, limit=limit, offset=offset)
    return [ConversationResponse(
        session_id=c.session_id,
        user_id=c.user_id,
        title=c.title,
        messages=[],
        created_at=c.created_at,
        updated_at=c.updated_at,
        metadata=c.metadata
    ) for c in conversations]

@app.post("/api/conversations/search", response_model=list[ConversationResponse], tags=["对话管理"])   # 搜索对话接口
async def search_conversations(req: ConversationSearchRequest):
    """搜索对话（按标题或消息内容）"""
    conversations = await memory_service.search_conversations(
        query=req.query,
        user_id=req.user_id or "",
        limit=req.limit
    )
    return [ConversationResponse(
        session_id=c.session_id,
        user_id=c.user_id,
        title=c.title,
        messages=[],
        created_at=c.created_at,
        updated_at=c.updated_at,
        metadata=c.metadata
    ) for c in conversations]

@app.get("/api/conversations/{session_id}", response_model=ConversationResponse, tags=["对话管理"])   # 获取单个会话详情接口
async def get_conversation(session_id: str):
    """获取单个会话详情（继续对话）"""
    conversation = await memory_service.get_conversation(session_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationResponse(
        session_id=conversation.session_id,
        user_id=conversation.user_id,
        title=conversation.title,
        messages=[MessageResponse(
            id=m.id,
            session_id=m.session_id,
            role=m.role,
            content=m.content,
            timestamp=m.timestamp,
            metadata=m.metadata
        ) for m in conversation.messages],
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        metadata=conversation.metadata
    )

@app.post("/api/conversations/{session_id}/messages", response_model=MessageResponse, tags=["对话管理"])   # 添加消息到会话接口
async def add_message(session_id: str, req: MessageRequest):
    """添加消息到会话（存储对话）"""
    message = await memory_service.save_message(
        session_id=session_id,
        role=req.role,
        content=req.content,
        message_id=req.message_id or "",
        metadata=req.metadata
    )
    return MessageResponse(
        id=message.id,
        session_id=message.session_id,
        role=message.role,
        content=message.content,
        timestamp=message.timestamp,
        metadata=message.metadata
    )

@app.put("/api/conversations/{session_id}/title", tags=["对话管理"])   # 更新会话标题接口
async def update_conversation_title(session_id: str, title: str):
    """更新会话标题"""
    success = await memory_service.update_conversation_title(session_id, title)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"success": True, "message": "Title updated"}

@app.delete("/api/conversations/{session_id}", tags=["对话管理"])
async def delete_conversation(session_id: str):
    """删除整个会话"""
    success = await memory_service.delete_conversation(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"success": True, "message": "Conversation deleted"}

@app.delete("/api/conversations/{session_id}/messages/{message_id}", tags=["对话管理"])   # 删除单条消息接口
async def delete_message(session_id: str, message_id: str):
    """删除单条消息"""
    success = await memory_service.delete_message(session_id, message_id)
    if not success:
        raise HTTPException(status_code=404, detail="Message not found")
    return {"success": True, "message": "Message deleted"}

@app.get("/api/health", tags=["系统管理"])      # 健康检查接口
async def health():
    """健康检查"""
    return {"status": "ok", "service": "AgentKnowledgeHub"}


@app.get("/api/health/ready", tags=["系统管理"])
async def readiness():
    """Dependency readiness without invoking billable model endpoints."""
    if not vector_store or not memory_service or not document_registry or not document_coordinator:
        raise HTTPException(status_code=503, detail="Application dependencies are not initialized")
    try:
        await vector_store.get_stats()
        await knowledge_graph.get_stats()
    except Exception:
        raise HTTPException(status_code=503, detail="A required datastore is unavailable") from None
    if not settings.has_usable_llm_key:
        raise HTTPException(status_code=503, detail="Chat provider configuration is incomplete")
    return {"status": "ready", "providers": {"chat": settings.chat_config.provider, "embedding": settings.embedding_config.provider}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)   # 启动API服务
    print(f"API service is running on {settings.api_host}:{settings.api_port}")
