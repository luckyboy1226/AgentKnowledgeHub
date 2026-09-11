# AgentKnowledgeHub 20 文档 / 60 问题 Benchmark

## 内容
- `benchmark_documents.json`：20 份固定企业文档、实体、关系、关键事实与干扰设计。
- `benchmark_questions.json`：60 道固定问题及 deterministic ground truth。
- `documents/`：20 个可直接用于真实上传的 UTF-8 `.txt` 文件。
- `benchmark_manifest.json`：规模与运行约束摘要。

## 题型
- Single-hop factoid：12
- Multi-hop relation：12
- Constraint / aggregation：12
- Distractor / conflict：12
- Abstention / unanswerable：12

## 设计原则
这是用于工程机制验证和秋招项目量化的中小规模合成评测，不代表生产场景的普遍性能。
Vector RAG 与 GraphRAG 应共用同一 corpus、同一 Query Plan、同一模型参数和同一最终 context 上限。
不得为了让 GraphRAG 获胜而删题、改标准答案或只保留最好的一次 run。

`relations` 和 `expected_relation_path` 是人工 benchmark ground truth。
真实系统仍应从 `content` 文档正文执行自己的知识抽取流程。

## 推荐真实运行安全边界
1. 每次 run 使用唯一 logical key：`benchmark-<run_id>-<document-name>`。
2. 只把本轮实际上传返回的 `document_id` 加入 `EvaluationScope`。
3. `vector_only` 必须保持 `graph_calls == 0`。
4. 评测结束后只按本轮实际 `document_id` 调用版本化 DELETE。
5. 禁止 collection-wide delete、volume delete、`delete_by_doc_id` legacy 路径和 `delete_by_source` legacy 路径。

## 当前规模
- 文档：20
- 问题：60
- 每类：12
