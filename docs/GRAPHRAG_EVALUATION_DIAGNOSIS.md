# S4.1：GraphRAG 对比评测诊断与优化设计

## 边界与不可变基线

本文件只分析真实 run `s4-20260910T152650Z` 的本地结果：
`.runtime/evaluation/s4-20260910T152650Z/results.json`。不改写该目录中的 JSON、CSV 或
Markdown，不重新评分，也不重新调用模型。

该 run 使用四份合成文档、12 道题和经验证的四个 `document_id` allowlist。两种模式均完成
12 题；`vector_only` 的 Neo4j 调用数为 0，所有 140 个返回 source 均在 allowlist 内。

| 指标 | vector_only | graph_rag |
|---|---:|---:|
| 问题级正确率（原 scorer） | 75.0% | 33.3% |
| 关键事实命中率 | 97.2% | 86.1% |
| 来源命中率 | 83.3% | 58.3% |
| 多跳题正确率 | 100% | 0% |
| 平均延迟 | 9.12s | 13.16s |
| 图谱参与率 | — | 91.7% |

这只是小型、单次、合成中文语料的机制基线，不代表生产数据上的普遍结论。它确实表明：**在
本次配置下，GraphRAG 没有胜出，且在多跳题上更差。** 后续 S4.3 必须保留此 run，不得覆盖、
删除或用新分数替换它。

### 可观察性边界

`results.json` 保留了 answer、sanitized sources、vector/graph context 数和计数性 reasoning，
但基于安全结果格式，**没有保存原始图谱记录、完整 prompt、每跳节点序列或
`EvaluationQueryPlan`**。因此下列“检索 miss”只在来源/上下文计数和调用路径共同支持时标为
中等证据；不能声称已经直接看见某一条未返回的 Neo4j relationship。

## 调用路径证据

当前 scoped `graph_rag` 的实际路径为：

```text
shared intent/rewrite → VectorStore.search(scope)
                     → KnowledgeGraphService.get_neighbors(scope, hops=2)
                     → _hybrid_rerank → _generate_answer
```

- `KnowledgeGraphService.get_neighbors` 使用无向模式
  `(start)-[rels*1..2]-(neighbor)`，只返回 `start`、`target`、`[type(rel)]`；多跳记录的
  `document_id`、version、source 使用 relationship 列表中的 `head(...)`。
- `QAAgent._graph_retrieve` 将该 record 直接 `str(record)` 放入 context，而非保留每跳的
  subject、predicate、object、direction、evidence key 与每条 provenance。
- `_hybrid_rerank` 对任何 graph context 固定乘以 `1.2`，随后只按分数截取前 8 条。
- 原评分器把 required keywords、source 集合和（拒答题）“没有 source”合成为一个
  `question_correct` 布尔值。

这些代码事实解释了为什么“图谱已参与”不等于“图谱证据足以、且被忠实使用”。

## 逐题诊断矩阵

`图谱来源/上下文` 仅摘录安全 results 中的 source 与计数；`主因`是基于 answer、source、
结果计数和上述代码路径的证据化归类，不是只根据最终分数猜测。

| ID | 类型 / 期望关键事实 | vector / graph 原评分 | graph 来源与上下文 | 诊断与主因 |
|---|---|---:|---|---|
| Q01 | 单跳；陈航、数据平台部、数据工程师 | ✓ / ✓ | 1 vector + 7 graph；org/projects | 正确；图谱真实参与但对单跳无可见增益。`insufficient_test_evidence` |
| Q02 | 单跳；陈航、Chroma | ✓ / ✓ | 2 vector + 6 graph；projects 为主 | 正确，但附带 Atlas/天枢信息；未造成评分错误。`insufficient_test_evidence` |
| Q03 | 多跳；北极星 → Atlas → 赵启 | ✓ / ✗ | 0 vector + 8 graph；collaboration/distractors，缺 projects | 图答案把责任泛化为天枢的两个部门/负责人，遗漏 Atlas。正确来源未进入结果；无向两跳关系压缩后也无法区分“依赖服务”与周边协作。主因 `graph_retrieval_miss`；次因 `graph_relation_ambiguity` |
| Q04 | 多跳；林岚/数据平台部，赵启/应用架构部 | ✓ / ✗ | 0 vector + 8 graph；仅 collaboration | 图答案正确给出共同负责人和两个部门，但明确拒绝逐人部门映射；org provenance 未进入图结果。主因 `graph_retrieval_miss`；次因 `context_merging_or_rerank_error` |
| Q05 | 多跳；北极星为天枢提供检索索引 | ✓ / ✗ | 0 vector + 8 graph；collaboration/distractors | 图答案改成“北极星依赖天枢”。原文本的 `提供检索索引` 被 `DEPENDS_ON` 泛化替代。主因 `graph_relation_ambiguity`；次因 `answer_generation_hallucination` |
| Q06 | 约束；天枢同时使用 Neo4j、Chroma | ✓ / ✓ | 0 vector + 8 graph；collaboration/projects | 正确；图谱能回答明确技术关系，但没有相较 vector 的增益。`insufficient_test_evidence` |
| Q07 | 约束多跳；陈航、北极星 | ✓ / ✗ | 0 vector + 8 graph；org/collaboration，缺 projects | 图答案语义上列出林岚和陈航及相应项目，required keywords 也出现；但预期 projects source 未命中、答案引用为编号而非可审计文件名。主因 `source_attribution_error`；原 `question_correct=false` 是部分 `scorer_false_negative` |
| Q08 | 多跳；赵启 → Atlas → 北极星 | ✓ / ✗ | 0 vector + 8 graph；org/projects/collaboration | 图答案把赵启直接说成与北极星存在 `DEPENDS_ON`，遗漏中间 Atlas。主因 `graph_relation_ambiguity`；次因 `answer_generation_hallucination` |
| Q09 | 干扰事实；PostgreSQL、Elasticsearch | ✓ / ✓ | 4 vector + 4 graph；distractors | 正确；图谱参与但没有必要优势。`insufficient_test_evidence` |
| Q10 | 否定；周宁不负责北极星 | ✗ / ✗ | 0 vector + 8 graph；org/distractors/projects | vector 答案明确“不负责”、并给出陈航负责/周宁参与晨星；原 scorer 只要求关键词 `否`，故为 `scorer_false_negative`。graph 答案选择“无法得出结论”：图中没有显式 `NOT_RESPONSIBLE` 或“唯一负责人”关系，不能仅由不存在的边推导否定。主因 `insufficient_test_evidence`，次因 `graph_relation_ambiguity` |
| Q11 | 无答案；南斗负责人 | ✗ / ✗ | 4 vector + 0 graph；四份 source | 两种答案均明确“未提及/无法回答”；scorer 要求拒答题 `sources` 为空，且 marker 集对“未提及”不完整，导致 false negative。主因 `scorer_false_negative` |
| Q12 | 无答案；公司 CEO | ✗ / ✗ | vector 0 graph 8；graph 为 org | 两种答案均表达“未提及”，但 scorer 未把该表达视为拒答；graph 还以非相关 org context 支撑拒答。主因 `scorer_false_negative`；次因 `context_merging_or_rerank_error` |

## 重点题结论

### Q03：Atlas 被部门/天枢事实替代

证据是 graph 结果为 0 vector + 8 graph，sources 全部为 `collaboration` 或 `distractors`，
而正确的 vector 回答使用 Atlas/赵启，预期来源是 `projects`。这不是 scorer 问题：图答案确实
没有回答 Atlas。主因是图谱检索精度/召回失败；无向、两跳、最多 50 条的邻居枚举再经固定权重
截断，使周边协作路径可以压过服务责任路径。

### Q05：提供检索索引被改写为依赖

graph 的安全 source 确实包含 `collaboration`，但答案从“提供检索索引”改为
`DEPENDS_ON`。这证明 source 在范围内不等于 relationship 语义保真。`get_neighbors` 不返回
可呈现的逐跳方向和中间节点，`str(record)` 又没有自然语言约束“不得把 PROVIDES_INDEX
推断为 DEPENDS_ON”；因此主因是关系歧义，最终生成是放大器。

### Q10：否定题不是单一失败

vector 的“不负责”与陈航负责、周宁参与晨星相符，却因硬编码 `否` 未出现而低分，属于评分器
误判。graph 的“无法得出结论”不应被武断标为幻觉：合成语料没有显式否定边或唯一性语义，
预期答案依赖闭世界假设。下一轮应分别测试“有直接否定证据”和“无证据时保守拒答”。

### Q11/Q12：拒答答案被评分规则吞没

两题答案均写出“未提及”；Q11 还写出“无法回答”。现有 scorer 同时要求 marker 命中且
`not sources`，所以有合理支持性 citation 的拒答也会失败；Q12 还缺少 `未提及` marker。
这是 scorer 设计缺陷，不是模型捏造事实的证据。

## 根因排序（按证据强度）

1. **关系表达丢失与歧义（强）**：无向 path、关系类型列表、单一 `head` provenance、
   `str(record)` context 与 Q05/Q08 的错误关系结论相互印证。
2. **无 provenance-aware 的固定图谱加权（强）**：Q03–Q08 多为 0 vector + 8 graph，正确
   vector 证据完全没有进入最终 prompt；任意 graph record 固定获得 1.2 倍权重。
3. **评分器耦合（强）**：Q10–Q12 的答案文本直接显示语义正确/可接受但
   `question_correct=false`；代码把语义、source、拒答 marker 合为一个布尔值。
4. **图谱检索覆盖不足（中）**：Q03 缺 projects、Q04 缺 org；results 能证明返回来源不足，
   但没有持久化原始 Neo4j records，不能断言是抽取、查询 plan、limit 或排序中的唯一环节。
5. **query plan/实体识别错误（证据不足）**：本 run 未保存 plan，不能把任何题直接归咎于 rewrite。

## S4.2a：确定性评分器修复方案

保持原始 report 不变；新 scorer 以独立版本号（例如 `deterministic-v2`）输出并并列保存：

- `answer_semantic_score`：required/forbidden 规则，不再与 source 规则相乘。
- `source_coverage_score`：期望来源集合、document_id/version 和引用展示格式分别计分。
- `abstention_score`：对无答案题接受 `无法回答`、`无法确定`、`信息不足`、`未提及`、
  `没有相关信息` 等受限表达；同时检查没有编造目标事实。
- `citation_behavior`：无答案题允许 source 作为“检索范围说明”，但不得将其伪装为目标事实来源。
- `overall_v2` 只作为透明派生值；报告旧分数、新分数、scorer version 和触发规则。

不采用 LLM-as-a-judge。所有否定/拒答表达使用显式 fixture 集合与 fake-only 单元测试。

### 实施状态

`deterministic-v2` 已实现为独立 scorer：否定题使用有限的否定表达和正向主张模式，拒答题使用
有限的拒答表达并拒绝“目标是某人”式编造。答案语义、来源覆盖和图谱参与均独立输出；因此 Q10、
Q11、Q12 可以获得可解释的重评分，但 Q03/Q05 的关系混淆仍保留为真实质量问题。历史原始
`results.json` 不会被覆盖，v2 只能写入新的 score-only 派生报告目录。

## S4.2b：最小 Graph context 优化候选

| 候选 | 最小实现 | 预期收益 | 风险 / 实现量 | 离线验证 |
|---|---|---|---|---|
| A：有向三元组模板 | 保留每跳 `subject --predicate--> object`、方向、evidence key、每条 source/version；以模板而非 `str(record)` 写入 prompt | 阻止 Q05/Q08 的关系泛化；来源可审计 | 中：需扩展图查询返回形状与 context formatter | fake 图返回多跳路径，断言 prompt 含方向、谓词、逐边 provenance，且不产生未给出的依赖关系 |
| B：provenance-aware 选择 | 仅提升与问题实体/谓词匹配的 graph edge；无直接支持时保留 vector 或省略 graph；删除全局 1.2 boost | 避免 Q03/Q04 的 8 条图上下文挤掉正确 vector | 中：需要可解释的 edge coverage 打分 | fake vector/graph 混合场景，断言错误/无关 graph 不进入前 8，vector-only 仍零图调用 |
| C：受限关系推断 | 为 `负责`、`依赖`、`提供索引`、`共同交付` 定义互不蕴含的自然语言模板；没有规则时只陈述原边 | 降低生成时将提供方改成依赖方 | 小：可与 A 同时实现 | fixture 检查禁止 `PROVIDES_INDEX → DEPENDS_ON`，允许原 predicate 忠实表达 |

推荐先实施 **A + C 的最小组合**：它直接修复最强证据的 path/provenance 丢失，不改变检索召回，
也最容易以 fake-only 测试锁定“不能把提供索引说成依赖”。随后再实施 B；否则调整权重会掩盖
关系表达错误而不能解释其来源。

### S4.2b 实施状态

已实施 A+C 的 scoped-evaluation 最小版本：Neo4j scoped reads 返回逐边 provenance，QA 只将
provenance 完整、allowlist 内、ready/current 的有向 edge 格式化为 graph context。关系谓词按
存储值原样输出，并明确显示路径方向和中间实体；无关、legacy、inactive 或缺 provenance 的
records 被 fail-closed 拒绝。评测专用 rerank 不再为 graph context 固定乘以 `1.2`。

这只改变内部评测 `graph_rag` 路径，未改变普通 `/api/qa/ask` 默认行为。它是为下一轮受控实验
降低关系混淆而做的机制修复，尚未重新运行真实评测，也不能宣称结果已有提升。

## 实施顺序与验收

1. **S4.2a**：实现 scorer v2 和离线 fixture；保留 `results.json` 原始 v1 分数，不做原地覆盖。
2. **S4.2b**：实现 A+C 的 graph context 格式，再以 fake-only 测试覆盖方向、谓词、provenance、
   source 和 vector-only 零图调用；随后评估是否需要 B。
3. **S4.3**：仅在用户再次逐项授权后，以新 run_id、新的 `s4-eval-*` 文档跑第二轮真实评测，
   精确删除新文档数据，并与第一轮并列比较。

验收不要求 GraphRAG 必须胜出。要求是：关系混淆减少、来源准确、scope 隔离仍 fail-closed、
vector-only 仍零图谱调用、评分规则可解释。第二轮若仍落后，必须如实保留该结果。
