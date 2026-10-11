# 路由式 ReAct 知识问答架构设计

## 目的与状态

本文记录目标在线问答架构，作为后续伪代码、接口拆分、灰度接入和测试
的共同依据。它描述的是**目标设计**，不是当前线上行为的声明。

当前 `QAAgent.answer()` 仍是 V1 路径：意图识别、Query Rewrite、向量检索、
图谱检索、启发式重排、生成答案。已有 Hybrid Retrieval V2 的 BM25、RRF、
rerank、Parent Expansion 和 Context Budget 是内部构件，默认没有接入
`/api/qa/ask`。本文定义未来如何在不破坏该默认行为的前提下接入。

相关既有设计：

- `docs/HYBRID_RETRIEVAL_V2_DESIGN.md`：候选契约、RRF、重排、父块扩展和预算。
- `docs/LOCAL_RERANK_DESIGN.md`：本地 BGE 重排的运行边界。
- `docs/PARENT_CHILD_CHUNK_DESIGN.md`：父子块数据契约和生命周期。
- `docs/GRAPH_EVIDENCE_TRACE_DESIGN.md`：图谱证据的可追溯性。

## 目标流程

```text
用户 Query
    |
    v
Rule Router ──(命中明确规则)──> RouteDecision
    |                                 |
    └──────(未命中/不确定)──> LLM Router
                                      |
                                      v
                                RouteDecision
                                  |          |
                    DIRECT_CHAT --+          +-- KNOWLEDGE_QA
                      |                              |
                      v                              v
                 Direct LLM                   Metadata Filter
                                                     |
                                                     v
                                             Query Rewrite
                                                     |
                                                     v
                                           ReAct Retrieval Agent
                                                     |
                               v
                          Thought / Decision
                     “还缺少哪类证据？”
                               |
                               v
                        Action: 选择工具
                     /         |          \
                  BM25       Vector       Graph
                     \         |          /
                      +--------+---------+
                               |
                               v
                        Observation
                     候选证据与诊断
                               |
                               v
                        Evidence Judge
                     “证据是否已足够？”
                          |           |
                         否           是
                          |           |
                          +---循环----+----> Candidate Pool
                                                    |
                                                    v
                                               RRF Fusion
                                                    |
                                                    v
                                                BGE Rerank
                                                    |
                                                    v
                                      Default Parent Expansion
                                                    |
                                                    v
                                              Context Budget
                                                    |
                                                    v
                                          Answer Generation
                                                    |
                                                    v
                                           Answer + Citation
```

`ReAct Retrieval Agent` 只负责检索规划和证据充分性循环；它不直接产出面向
用户的答案，也不允许绕开统一候选池、范围过滤或最终上下文预算。

## 路由层

### RouteDecision

```python
class Route(str, Enum):
    DIRECT_CHAT = "direct_chat"
    KNOWLEDGE_QA = "knowledge_qa"

@dataclass(frozen=True)
class RouteDecision:
    route: Route
    reason_code: str                 # 仅枚举值，不记录 prompt 推理文本
    confidence: float | None
    rule_matched: bool
```

### Rule Router

规则路由器是低延迟、可审计的优先层。规则命中时直接返回决定，不调用 LLM
Router。建议第一版只使用保守的确定性规则：

| 条件 | 决策 | 示例 reason_code |
|---|---|---|
| 空输入、仅问候、闲聊、写作或翻译请求 | `DIRECT_CHAT` | `social_or_creative` |
| 明确要求“根据知识库/文档/制度/附件回答” | `KNOWLEDGE_QA` | `explicit_knowledge_request` |
| 包含内部文档名、已知实体 ID、错误码或受控术语 | `KNOWLEDGE_QA` | `known_corpus_anchor` |
| 需要当前会话记忆但不需要文档事实 | `DIRECT_CHAT` | `memory_only` |

规则不能因“没有匹配”而默认走 Direct LLM；未命中应交给 LLM Router，避免
遗漏隐含知识库问题。规则表必须版本化、可单测，并在日志中只保留规则 ID。

### LLM Router

LLM Router 仅在规则未命中或规则存在冲突时运行。其输出必须限制为
`direct_chat` 或 `knowledge_qa` 的结构化枚举；解析失败、安全降级或置信度
低于阈值时选择 `KNOWLEDGE_QA`，以“可说明资料不足”替代无依据回答。

## KNOWLEDGE_QA 路径

### 1. Metadata Filter（检索前置范围收缩）

进入 `KNOWLEDGE_QA` 后，任何文本、向量或图谱检索之前，先运行元数据过滤。
它根据请求的访问主体、租户/知识库、会话已选文档、文档类型、状态、时间范围、
标签、业务域和可识别的文档锚点，构造统一的 `RetrievalScope`。该步骤的目的不是
对正文做相关性排序，而是尽早把候选库缩小到“允许且可能相关”的文档集合。

```python
@dataclass(frozen=True)
class RetrievalScope:
    tenant_id: str
    knowledge_base_ids: frozenset[str]
    allowed_document_ids: frozenset[str] | None
    metadata_filters: Mapping[str, tuple[str, ...]]
    lifecycle: str = "ready_current"
    source: str = "metadata_filter"

@dataclass(frozen=True)
class MetadataFilterResult:
    scope: RetrievalScope
    matched_document_count: int | None  # 无权或不可计数时为 None
    reason_codes: tuple[str, ...]
```

过滤器必须先应用不可放宽的访问控制和租户隔离，再应用用户筛选条件和 Query 中的
显式元数据线索。例如“查询 2026 年 Q3 的运维手册”可以约束 `document_type`、
`period` 和标签；但模型推测出的元数据条件不得替代显式 ACL。BM25、Vector、Graph、
Parent Lookup 都必须接收同一个 `RetrievalScope`，而非各自拼接过滤条件。

空的受限范围是 fail-closed：直接进入“当前范围内没有可用资料”的受限回答，或仅
在产品显式允许时请求用户放宽筛选；绝不回退为全库检索。元数据过滤失败时同样不能
放宽 ACL，可退化为仅使用已验证的请求级范围。

### 2. Query Rewrite

输入原始 Query、`RetrievalScope` 和可公开的上下文。输出 `RetrievalPlan`：

```python
@dataclass(frozen=True)
class RetrievalPlan:
    original_query: str
    queries: tuple[str, ...]         # 1–3 条，去重且保留原问题语义
    entities: tuple[str, ...]
    keywords: tuple[str, ...]
    intent: QueryIntent
    scope: RetrievalScope
```

改写只服务检索，原始问题始终用于最终回答。空的受限范围是 fail-closed：不执行
任何无范围回退检索。

### 3. ReAct Retrieval Agent

每轮由一个受控决策器根据已有 Observation 选择动作。模型可以提出动作，
但执行器必须校验动作模式、参数范围和总预算。

```python
class RetrievalActionType(str, Enum):
    SEARCH_BM25 = "search_bm25"
    SEARCH_VECTOR = "search_vector"
    SEARCH_GRAPH = "search_graph"
    REWRITE_QUERY = "rewrite_query"
    EXPAND_PARENT = "expand_parent"
    FINISH = "finish"

@dataclass(frozen=True)
class RetrievalAction:
    type: RetrievalActionType
    query_index: int | None = None
    top_k: int | None = None
    rationale_code: str = ""        # 枚举化，例如 exact_id_needed
```

允许的工具与使用倾向：

| 工具 | 擅长内容 | 不应承担的职责 |
|---|---|---|
| BM25 | 精确名称、ID、错误码、版本、短语 | 模糊语义召回 |
| Vector | 语义近似、同义改写、长描述 | 精确实体关系证明 |
| Graph | 实体关系、方向明确的事实、多跳路径 | 替代原文细节或猜测关系 |
| Parent Lookup | 已命中 Child 缺少上下文时补全完整段落 | 扩大为不受约束的父文档搜索 |

每次工具调用都产生 `Observation`，仅含排名候选、稳定标识、来源、文档/版本
provenance、分数、召回器类型和安全诊断。不得将完整 prompt、密钥、绝对路径或
未经处理的内部元数据写入 trace。

### 4. Evidence Judge 与循环

Evidence Judge 不依据“检索到了几条”判断结束，而验证问题所需主张是否被证据
覆盖。建议输出：

```python
@dataclass(frozen=True)
class EvidenceAssessment:
    sufficient: bool
    coverage: float                  # 0–1，辅助指标而非唯一门槛
    missing_aspects: tuple[str, ...] # 如 definition / date / relation / comparison
    next_strategy: str | None        # 枚举值；不足时必填
    stop_reason: str | None          # 足够或预算终止时必填
```

循环策略：

1. 首轮可并行执行 BM25、Vector、Graph，或由意图决定最小工具集合。
2. 若某个已命中 Child 的事实片段相关但上下文不完整，优先执行受约束的
   `EXPAND_PARENT`：只可从该 Child 记录的精确 `parent_chunk_id` 回溯，并将父块
   作为补充 Observation 纳入当前轮次；它不是一次新的全库检索。
3. 其他证据不足场景，优先针对 `missing_aspects` 选择未使用的检索器；其次改写
   Query；最后以不同查询补充检索。
4. 任一轮候选都累积到 Candidate Pool，不能因下一轮失败而丢弃先前有效证据。
5. 达到充分条件时 `FINISH`；达到硬预算时也必须停止，并将不足信息带给生成器。

硬限制建议首版配置化：最多 3 轮决策、每检索器最多 2 次、最多 8 次工具调用、
总检索超时 12 秒；`EXPAND_PARENT` 最多 2 次且计入 8 次总工具调用。超时、工具失败或结构化输出非法时记录安全 reason code，继续
使用已获得候选；没有候选时进入“资料不足”回答，而不是编造答案。

## 候选融合到答案

ReAct 循环完成后，Candidate Pool 必须沿用 Hybrid V2 的受控后处理：

```text
Candidate Pool
  -> eligibility / RetrievalScope recheck
  -> RRF Fusion (Top 30)
  -> BGE Rerank (Top 12；失败则保留 RRF 顺序)
  -> Default Parent Expansion（默认上下文补充）
  -> Context Budget (最多 8 个完整证据单元、估算 6000 tokens)
  -> Answer Generation
```

### Parent Expansion 的两种模式

| 模式 | 触发时机 | 输入 | 输出与目的 |
|---|---|---|---|
| `react_on_demand` | ReAct Judge 发现已命中 Child 的内容不完整 | 已命中 candidate ID 与其精确 parent ID | 在检索循环中取得父块补充 Observation，帮助 Judge 决定是否继续检索 |
| `default_context` | RRF / rerank 完成后，构建最终 Context | 排序后的 Child | 将对应父块作为完整上下文单元拼接/替代 Child，供答案生成阅读 |

两种模式共享同一安全解析器：只能使用 Child 自身保存的
`(document_id, document_version, parent_chunk_id)`，并且父块必须属于同一
`RetrievalScope`、`kind=parent`、`status=ready`、`is_current=true`。图谱候选不能
触发 Parent Lookup，也不参与默认父块扩展。

`react_on_demand` 是**检索期补全**：只有 Judge 标出 `context_incomplete`、
`missing_definition`、`missing_exception`、`missing_prerequisite` 等与当前 Child
相关的缺口才可选择；其 Observation 应保留触发它的 child/candidate ID。读取到的
父块与原 Child 一并留在 Candidate Pool，不能因父块存在就覆盖 Child 的检索来源和
排序证据。

`default_context` 是**生成期补充**：它不改变 RRF 或 rerank 排名，也不触发新的
ReAct 决策。对同一父块的多个命中 Child 去重后，将 Child 的命中点与父块正文合并
为一个 `FinalContext`；预算不足时优先选择能容纳的 Child fallback，不能截断父块。
这样既保留“为何命中”的可引用锚点，也给生成器足够的段落语境。

关键不变量：

- BM25、Vector、Graph 原始分数不相加；跨检索器排序只由 RRF 完成。
- RRF、rerank 分数分别保留，不做手工混分。
- 所有阶段重复执行生命周期和 `RetrievalScope` 校验；空范围不回退到全库。
- Parent Expansion 只接受准确的 `(document_id, document_version, parent_chunk_id)`；
  缺少、越权或过期父块时保留 child，不猜测邻近文本。
- 图谱关系按原始 `subject --predicate--> object` 和方向呈现，不能反向或泛化。
- Context Budget 选择完整证据单元，不截断父块以凑预算。

最终生成器接收原始 Query、受预算限制的 `FinalContext`、证据不足状态和引用编号。
它必须仅基于这些证据陈述事实；回答中每个可验证主张应带可解析的来源引用。

## 推荐编排伪代码

```python
async def answer_user_query(request: QARequest) -> QAResult:
    route = await decide_route(request.query, request.session_context)
    if route.route is Route.DIRECT_CHAT:
        return await direct_llm_answer(request, route)

    metadata = await filter_metadata_before_retrieval(request)
    if metadata.scope.allowed_document_ids == frozenset():
        return no_evidence_in_scope_answer(request.query, metadata)

    plan = await rewrite_query(request.query, metadata.scope)
    state = RetrievalState(plan=plan, candidate_pool=[], observations=[])

    for _ in range(settings.REACT_MAX_STEPS):
        assessment = await judge_evidence(state)
        if assessment.sufficient:
            break
        action = await choose_retrieval_action(state, assessment)
        if action.type is RetrievalActionType.FINISH:
            break
        observation = await execute_validated_action(
            action,
            state.plan,
            candidates=state.candidate_pool,  # parent lookup 只能引用已命中 Child
        )
        state = state.with_observation(observation)

    final_context = build_final_context(
        candidates=state.candidate_pool,
        plan=state.plan,
        # RRF -> rerank -> parent expansion -> complete-unit budget
    )
    return await generate_grounded_answer(
        question=request.query,
        contexts=final_context,
        evidence_assessment=assessment,
        route=route,
    )
```

实现时 `assessment` 必须在循环前初始化为“尚未评估”；`execute_validated_action`
不可直接将 LLM 产出的检索参数传给数据库或 Cypher 执行器。

## 建议模块边界

```text
python/routing/
  rule_router.py            # 确定性规则与版本
  llm_router.py             # 受限结构化路由
  models.py                 # RouteDecision
python/retrieval/
  metadata_filter.py        # 访问范围与文档元数据的前置收缩
  react_agent.py            # 状态机、预算、动作白名单
  evidence_judge.py         # 充分性评估
  action_executor.py        # BM25 / Vector / Graph 的安全适配
  candidates.py             # 既有统一候选契约
  fusion.py                 # 既有 RRF
  context_builder.py        # 既有 rerank/父块/预算
python/agents/
  qa_agent.py               # 编排入口或对上述模块的兼容适配
```

路由、元数据过滤、改写、ReAct 决策、Judge、检索执行和答案生成应以协议接口注入，方便用
fake provider 做确定性单元测试；不要让测试依赖真实 LLM、Neo4j 或向量库。

## 灰度、可观测性与验收

新增功能须由独立开关保护，例如 `ROUTED_REACT_QA_ENABLED=false`。关闭时继续走
当前 V1 行为；开关开启后也应可单独关闭 Rule Router、LLM Router、ReAct 和每种
检索器以便排障。先以内部 allowlist 或离线评测验证，再逐步放量。

每次请求记录安全 trace：route、规则/模型 reason code、动作序列、每工具耗时、
元数据过滤 reason code、过滤前后文档计数（若安全可计数）、候选数、Judge stop reason、
Parent Expansion 模式/次数、融合/重排/预算诊断和引用数。Query 可使用摘要指纹和
长度替代原文；不得记录答案全文、文档正文、凭据或绝对路径。

首版验收至少覆盖：

1. 闲聊和创作问题路由到 `DIRECT_CHAT`，不访问知识库。
2. 显式文档问题路由到 `KNOWLEDGE_QA`，元数据过滤先于所有检索器执行，且引用来自允许范围。
3. 空范围、元数据过滤失败与父块越权均 fail-closed，不能退回全库。
4. Judge 不足时能选择补充检索、改写或按已命中 Child 的 parent ID 补全，且始终受轮数和超时限制。
5. BM25 + Vector 相同 child 能在 RRF 合并；Graph 证据保持独立、关系方向正确。
6. 默认父块补充不改变 rerank 顺序；预算不足时保留 Child fallback。rerank 不可用、父块缺失、单个检索器失败均安全降级。
7. 生成答案在证据不足时明确说明不足，不能补全未经证据支持的事实。

## 非目标

本设计暂不引入自主写入、文档更新、背景任务、无限 ReAct 循环、未经校验的任意
Cypher、在线评测副作用或默认切换现有 API 路径。模型的 Thought 不需要也不应向
用户或持久化日志暴露；对外只提供可审计的动作与 reason code。
