# S4 第 1 阶段：Vector RAG 与 GraphRAG 对比评测设计

## 目标与边界

本评测比较 `vector_only` 与 `graph_rag`，不预设图谱一定更好。重点观察多跳关系、关系约束、来源可追溯性，同时如实记录图谱带来的延迟、空结果和失败率。数据集仅使用合成中文文本；本阶段不上传、不调用模型、不访问任何数据库。

当前 API QA 实际执行的是 `QAAgent.answer`：

```text
intent（Chat）→ rewrite（Chat）→ vector retrieval（Embedding + Chroma）
      → graph retrieval（Neo4j get_neighbors）→ hybrid rerank → final answer（Chat）
```

`QAAgent._graph_retrieve` 使用 rewrite 提取的实体调用参数化、current/legacy-aware 的
`KnowledgeGraphService.get_neighbors`。`GraphRAGPipeline` 另有实体链接、路径及社区摘要能力，
但当前不在 `/api/qa/ask` 的运行路径中；S4 第一版不把它作为额外变量。

## 公平对照规则

| 维度 | 规则 |
|---|---|
| Provider 与参数 | 两种模式使用同一 Chat/Embedding provider、模型、`temperature=0`、Chat timeout 60 秒、Chat retry 1 次、Embedding timeout/retry 配置。 |
| 预处理 | 每题只生成一次 intent/rewrite 快照，并在两种模式中原样复用；这样实体、检索 query 与其模型成本相同。 |
| 向量检索 | 两种模式对每个共享 query 都以相同 `top_k=5` 调用 Chroma；S3 的 current/legacy 过滤保持开启。 |
| 最终生成 | 两种模式使用相同的最终回答提示、上下文上限 8、问题文本、超时、重试和模型参数。 |
| Memory | 评测 factory 必须注入 `memory_service=None`，不读取或写入 profile、短期/长期记忆或 checkpoint，以避免会话状态成为变量。 |
| 数据范围 | 评测 factory 只能从四次 S3 上传响应获得 UUID `document_id`，构造非空、已验证的 `EvaluationScope` allowlist。没有 scope、空 scope、格式非法或未验证的 scope 必须拒绝运行，绝不退回全库。 |
| vector_only | 禁止调用 Neo4j、禁止构造 graph context，最终 prompt 只能包含 vector context。 |
| graph_rag | 保留相同 vector context，额外调用现有 `get_neighbors`；最终 prompt 可包含 vector + graph context。 |

因此，唯一允许影响两种回答的变量是是否增加图谱上下文。评测 runner 应使用共享
`EvaluationQueryPlan`（intent、一个规范化 query、entities），而不是让两个模式独立 rewrite；
这是一项检索对照，不是两次随机 rewrite 的对照。

### 文档范围隔离

真实评测在四份合成文档全部入库并返回 `document_id` 后，才构造
`EvaluationScope(run_id, allowed_document_ids)`。该对象不是公开 API 参数，普通
`/api/qa/ask` 不传 scope，因此其兼容行为不变。

- Chroma 保持有界 over-fetch，但在应用层严格二次过滤：只有 `document_id` 位于 allowlist、
  且具有 `document_version` 的 ready/current 向量可进入 context。legacy、缺失 provenance、
  非 allowlist 和检索不足均不得触发全库 fallback。
- Neo4j 在参数化 Cypher 中为路径的每条 evidence relationship 传入
  `rel.document_id IN $allowed_document_ids` 与 ready/current 条件；应用层再次校验返回的
  `document_id` 和版本。scoped 图查询永不接受 legacy 无 provenance 关系，也不会在空结果时查询全图。
- `vector_only` 仍不调用任何 KnowledgeGraph 方法。最终 prompt、sources、context 计数只消费
  过滤后的结果；报告额外保存允许 ID 数、两种 scope 拒绝计数和 `scope_verified`，不保存正文或秘密。

## 拟议固定合成语料

后续真实运行时将以 `s4-eval-*` logical key 上传以下四份文本，并在完成后只按其验证过的
`document_id` 精确删除：

| 来源标识 | 合成事实 |
|---|---|
| `s4-eval-org.txt` | 星河云智公司有数据平台部、应用架构部和运营部。林岚是数据平台部负责人；陈航是数据工程师，隶属数据平台部；赵启是应用架构部负责人；周宁是运营部分析师。 |
| `s4-eval-projects.txt` | 陈航负责北极星检索项目。北极星使用 Chroma，依赖 Atlas 事件服务。赵启负责 Atlas 事件服务。 |
| `s4-eval-collaboration.txt` | 数据平台部与应用架构部共同交付天枢知识平台。林岚和赵启共同负责天枢。天枢使用 Neo4j 与 Chroma；北极星为天枢提供检索索引。 |
| `s4-eval-distractors.txt` | 晨星报表项目使用 PostgreSQL 与 Elasticsearch。周宁参与晨星报表的数据分析；晨星与北极星、天枢不存在依赖关系。 |

其中“北极星 → Atlas → 赵启”和“人员 → 部门 → 天枢”构成至少两跳/三跳路径；晨星、
周宁和 PostgreSQL/Elasticsearch 是干扰事实。南斗项目和 CEO 则不在语料中。

## 固定问题集

关键词匹配采用大小写无关、允许常见同义表达；每题还定义禁止事实，避免仅靠长回答命中。

| ID | 问题 | 类型 | 期望关键事实 | 至少命中来源 | 图谱预期优势 |
|---|---|---|---|---|---|
| Q01 | 陈航在什么部门担任什么职位？ | 单跳 | 数据平台部；数据工程师 | `s4-eval-org.txt` | 否 |
| Q02 | 北极星检索项目由谁负责，使用什么技术？ | 单跳 | 陈航；Chroma | `s4-eval-projects.txt` | 否 |
| Q03 | 北极星依赖的服务由谁负责？ | 多跳 | Atlas；赵启 | `s4-eval-projects.txt` | 是 |
| Q04 | 天枢知识平台的共同负责人分别来自哪个部门？ | 多跳 | 林岚/数据平台部；赵启/应用架构部 | `s4-eval-org.txt`、`s4-eval-collaboration.txt` | 是 |
| Q05 | 北极星如何与天枢知识平台关联？ | 多跳 | 北极星提供检索索引 | `s4-eval-collaboration.txt` | 是 |
| Q06 | 哪个项目同时使用 Neo4j 和 Chroma？ | 聚合/约束 | 天枢知识平台 | `s4-eval-collaboration.txt` | 中等 |
| Q07 | 谁同时满足“数据平台部成员”和“项目负责人”？ | 聚合/约束 | 陈航；北极星 | `s4-eval-org.txt`、`s4-eval-projects.txt` | 是 |
| Q08 | 赵启负责的服务与哪个项目存在依赖关系？ | 多跳 | Atlas；北极星 | `s4-eval-projects.txt` | 是 |
| Q09 | 晨星报表项目使用哪些技术？ | 干扰事实 | PostgreSQL；Elasticsearch | `s4-eval-distractors.txt` | 否 |
| Q10 | 周宁是否负责北极星检索项目？ | 干扰/否定 | 明确否定；周宁仅参与晨星分析 | `s4-eval-org.txt`、`s4-eval-distractors.txt` | 中等 |
| Q11 | 南斗项目的负责人是谁？ | 无法回答 | 明确说明语料没有该事实 | 无 | 否 |
| Q12 | 星河云智公司的 CEO 是谁？ | 无法回答 | 明确说明语料没有该事实 | 无 | 否 |

## 指标与确定性评分

LLM 不是本阶段的唯一裁判。每题在 fixture 中维护 `required_keywords`、`forbidden_keywords`、
`expected_sources`、`requires_abstention` 和 `graph_advantage_expected`。

- **运行成功率**：HTTP/runner 成功结果数 ÷ 题数；超时、异常、非 200 分别统计。
- **关键事实命中率**：命中的 required keywords ÷ required keywords 总数。
- **问题级正确率**：所有 required 命中、没有 forbidden、且来源规则通过的问题比例。
- **来源命中率**：正例题至少有一个期望来源标识，且返回安全 document/version；无答案题不应伪造来源。
- **多跳正确率**：Q03/Q04/Q05/Q07/Q08 的问题级正确率。
- **无答案拒答正确率**：Q11/Q12 明确表示信息不足/未找到，且不臆造实体或职位。
- **图谱参与率**：仅 `graph_rag`；记录图谱调用尝试率、非空 graph context 率和最终 prompt 的 graph context 数。不可用/空图谱不能伪装为参与。
- **耗时**：每模式计算平均、P50、P95（小样本采用 nearest-rank，并报告样本数）。
- **模型调用次数**：记录每题共享预处理、每模式回答、Embedding 查询的计数；仅记录方法/结果/耗时，不记录 prompt 或密钥。

报告必须同时展示每题，而不是只展示平均值；尤其保留 graph 不增益、返回空图、失败或延迟升高的样本。

## 后续结果格式

运行目录固定为 `.runtime/evaluation/<run_id>/`，且由 Git 忽略。每条题目结果只保存：

```text
run_id, mode, question_id, http_status, elapsed_ms,
answer, sanitized_sources, vector_context_count, graph_context_count,
model_call_counts, deterministic_score, failure_summary
```

`sanitized_sources` 只保留 source 文件名、document_id、document_version、retrieval_type 和分数；
不保存 API key、Authorization、环境变量、完整连接串、绝对路径、provider 原始响应或检索上下文全文。
每个 run 产出 `results.json`、`results.csv` 和 `summary.md`；汇总包含两个模式的指标、问题明细、
失败样本和未达显著优势的结论。

## 真实运行的授权边界

下一阶段在真实调用前必须向用户逐项列出：四份合成文档全文、12 个问题全文、`s4-eval-*`
logical key 前缀、预计调用量和精确删除范围，然后等待一次明确授权。四份上传操作完成后还必须
记录其精确 UUID `document_id`，构造并离线/运行时验证非空 `EvaluationScope`；没有该验证不允许
开始任一 QA 模式。

当前方案按每文档一个 chunk 估算：入库约 4 次 Chat 抽取与 4 次 Embedding；共享预处理最多
24 次 Chat（12 intent + 12 rewrite）；两模式最终回答约 24 次 Chat；两模式各一次向量检索约
24 次 Embedding。实际重试会单独计数，且在授权说明中更新为上限。不得在没有新授权时调用。

## 局限性

- 小型合成语料只能验证机制，不代表生产语料质量或真实业务分布。
- 虽然 temperature 为 0，外部模型与服务仍可能产生非确定性或瞬时失败。
- GraphRAG 可能提升多跳/约束题，却带来 Neo4j 查询、上下文膨胀和更高延迟。
- 旧 legacy 数据仍可被当前兼容检索返回；评测使用独立 synthetic source 标识，并在报告中单列非评测来源。

## S4 第 2 阶段：已实现的离线框架

已实现的 `scripts/run-rag-eval.py --offline --mode both` 仅构造内存 fake Chat、VectorStore 和
KnowledgeGraph，并通过 `QAAgent` 的内部受限 retrieval mode 运行 12 道题。`vector_only` 从代码
分支跳过图谱服务，`graph_rag` 才调用图谱；每题只构造一次 `EvaluationQueryPlan`，供两个模式共享。
评测 factory 强制 `memory_service=None`，所以不会读取或写入会话、checkpoint 或用户记忆。

离线输出写入被忽略的 `.runtime/evaluation/<run_id>/`，包含 JSON、CSV 和 Markdown。它仅证明
模式隔离、共享 plan、评分和报告格式正确；**fake-only 结果不能用于宣称 GraphRAG 优于 Vector RAG**。
脚本不带 `--offline` 时会明确拒绝真实运行，不会初始化真实 Provider 或数据库连接。

## 已完成的最小实现清单

1. 给 `QAAgent` 增加受限的 `vector_only`/`graph_rag` retrieval mode，并让 vector-only 从代码层不调用图服务。
2. 实现共享 `EvaluationQueryPlan`、memoryless evaluation factory 与 provider 调用计数包装器。
3. 实现 fixture、runner、确定性 scorer 和 JSON/CSV/Markdown 报告器。
4. 用 fake Chat、Embedding、VectorStore、KnowledgeGraph 编写离线单元测试，证明模式隔离、相同预处理、评分和敏感字段脱敏。
5. 仅在单独授权后，运行上述合成语料的真实入库、双模式评测和精确删除。

## S4.2a：版本化确定性评分

原始真实 run 的 `deterministic_score` 保持为 v1，绝不原地修改。新的
`deterministic-v2` 将答案语义、来源覆盖、拒答质量和引用行为分离为独立字段：
`answer_semantic_score`、`source_coverage_score`、`abstention_score`、
`citation_behavior` 与透明的 `overall_v2`。来源不足不再把语义正确的答案判为错误；
无答案题允许来源用于描述检索范围，但不允许编造目标事实。

历史 run 只能通过 `python scripts/run-rag-eval.py --rescore <results.json>` 生成相邻的
`<run_id>-rescored-v2/` 派生目录。派生报告不复制模型回答全文，保留 scorer 版本和 v1/v2 对比，
因此不会篡改不可变 baseline，也不能将真实关系混淆解释为评分问题。

## S4.2b：有向图谱证据与关系保真

内部 scoped `graph_rag` 查询现在只返回 `evidence_edges`：每条 edge 都包含存储方向的
`subject --predicate--> object`、路径遍历方向、`document_id`、版本、安全 source 和
`evidence_key`。缺任何一跳 provenance、包含 legacy/非 allowlist document 或非 ready/current
edge 的 record 会在进入 prompt 前被拒绝。普通未 scoped 的 API 查询仍使用旧返回形状。

评测 prompt 使用固定三元组模板，严格按已存 predicate 陈述；它不会将 `PROVIDES_INDEX` 改写为
`DEPENDS_ON`，也不会在缺少直接 edge 时补造关系。评测路径还移除了固定的 graph `1.2` 倍加权，
让 provenance-complete graph context 与 vector context 按原始分数竞争；未引入题目或答案硬编码。
