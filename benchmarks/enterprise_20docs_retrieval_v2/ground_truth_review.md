# Retrieval Ground Truth Review

Benchmark: `enterprise_20docs_retrieval` v`2.0.0`

| Question | Category | Answerable | Relevant logical documents | Derivation | Manual review |
|---|---|---|---|---|---|
| Q01: 陈航属于哪个部门，担任什么职位？ | single_hop | True | D02 | expected_sources basename -> benchmark_documents.id | False |
| Q02: 数据平台部负责人是谁？ | single_hop | True | D01 | expected_sources basename -> benchmark_documents.id | False |
| Q03: Atlas 事件服务由谁负责？ | single_hop | True | D08 | expected_sources basename -> benchmark_documents.id | False |
| Q04: 北极星检索平台当前使用哪个向量数据库？ | single_hop | True | D07 | expected_sources basename -> benchmark_documents.id | False |
| Q05: 天枢知识平台使用哪两种核心存储技术？ | single_hop | True | D09 | expected_sources basename -> benchmark_documents.id | False |
| Q06: 猎户座权限服务由谁负责？ | single_hop | True | D11 | expected_sources basename -> benchmark_documents.id | False |
| Q07: 晨星报表系统的项目负责人是谁？ | single_hop | True | D15 | expected_sources basename -> benchmark_documents.id | False |
| Q08: 晨星报表系统使用哪两种数据技术？ | single_hop | True | D15 | expected_sources basename -> benchmark_documents.id | False |
| Q09: 星图推荐引擎当前在线向量检索使用什么？ | single_hop | True | D13, D19 | expected_sources basename -> benchmark_documents.id | False |
| Q10: 河图数据湖采用哪些核心数据技术？ | single_hop | True | D14 | expected_sources basename -> benchmark_documents.id | False |
| Q11: 流光监控平台使用什么监控与可视化工具？ | single_hop | True | D16 | expected_sources basename -> benchmark_documents.id | False |
| Q12: 曙光客户门户由谁负责？ | single_hop | True | D12 | expected_sources basename -> benchmark_documents.id | False |
| Q13: 北极星检索平台依赖的事件服务由谁负责？ | multi_hop | True | D07, D08 | expected_sources basename -> benchmark_documents.id | False |
| Q14: 天枢知识平台依赖的权限服务由谁负责？ | multi_hop | True | D09, D11 | expected_sources basename -> benchmark_documents.id | False |
| Q15: 曙光客户门户依赖的知识平台由哪两位共同负责人负责？ | multi_hop | True | D12, D09 | expected_sources basename -> benchmark_documents.id | False |
| Q16: 华远能源使用的门户所依赖的权限服务由谁负责？ | multi_hop | True | D20, D12, D11 | expected_sources basename -> benchmark_documents.id | False |
| Q17: 陈航负责的项目依赖的服务使用什么消息队列？ | multi_hop | True | D07, D08 | expected_sources basename -> benchmark_documents.id | False |
| Q18: 沈舟负责的数据湖向哪个报表系统提供数据？ | multi_hop | True | D14 | expected_sources basename -> benchmark_documents.id | False |
| Q19: 星图推荐引擎负责人属于哪个部门？ | multi_hop | True | D13, D04 | expected_sources basename -> benchmark_documents.id | False |
| Q20: 云桥 API 网关负责人属于哪个部门？ | multi_hop | True | D17, D03 | expected_sources basename -> benchmark_documents.id | False |
| Q21: 天枢知识平台从哪个项目获得检索索引，该项目由谁负责？ | multi_hop | True | D10, D07 | expected_sources basename -> benchmark_documents.id | False |
| Q22: E-2026-07-18 事件影响的检索平台由谁负责？ | multi_hop | True | D18, D07 | expected_sources basename -> benchmark_documents.id | False |
| Q23: 华远能源使用的知识平台依赖哪个权限服务？ | multi_hop | True | D20, D12, D09 | expected_sources basename -> benchmark_documents.id | False |
| Q24: 流光监控的服务中，哪一个同时也是北极星检索平台的依赖？ | multi_hop | True | D16, D07 | expected_sources basename -> benchmark_documents.id | False |
| Q25: 哪些人既是部门负责人，又是天枢知识平台的共同负责人？ | constraint | True | D01, D09 | expected_sources basename -> benchmark_documents.id | False |
| Q26: 谁既属于安全运营部，又负责猎户座权限服务？ | constraint | True | D05, D11 | expected_sources basename -> benchmark_documents.id | False |
| Q27: 哪个项目同时使用 Neo4j 和 Chroma？ | constraint | True | D09 | expected_sources basename -> benchmark_documents.id | False |
| Q28: 哪个服务同时使用 Redis 和 PostgreSQL，并为天枢知识平台提供鉴权？ | constraint | True | D11 | expected_sources basename -> benchmark_documents.id | False |
| Q29: 谁既属于数据平台部，又负责北极星检索平台？ | constraint | True | D02, D07 | expected_sources basename -> benchmark_documents.id | False |
| Q30: 谁既属于智能产品部，又负责星图推荐引擎？ | constraint | True | D04, D13 | expected_sources basename -> benchmark_documents.id | False |
| Q31: 哪个系统同时使用 PostgreSQL 和 Elasticsearch，而且负责人不是周宁？ | constraint | True | D15 | expected_sources basename -> benchmark_documents.id | False |
| Q32: 哪个项目当前使用 Milvus，同时在路线图中计划评估 Chroma？ | constraint | True | D13, D19 | expected_sources basename -> benchmark_documents.id | False |
| Q33: 哪些项目都依赖猎户座权限服务？ | constraint | True | D09, D12, D11 | expected_sources basename -> benchmark_documents.id | False |
| Q34: 负责 Atlas 事件服务和云桥 API 网关的两个人属于同一个哪个部门？ | constraint | True | D03, D08, D17 | expected_sources basename -> benchmark_documents.id | False |
| Q35: 哪个项目既受到 E-2026-07-18 事件影响，又向天枢知识平台提供检索索引？ | constraint | True | D18, D10 | expected_sources basename -> benchmark_documents.id | False |
| Q36: 华远能源的知识服务入口同时使用哪两个后端能力完成问答和认证？ | constraint | True | D20 | expected_sources basename -> benchmark_documents.id | False |
| Q37: 周宁是否负责北极星检索平台？如果不是，谁负责？ | distractor | True | D05, D07 | expected_sources basename -> benchmark_documents.id | False |
| Q38: 晨星报表系统是否依赖北极星检索平台或天枢知识平台？ | distractor | True | D15 | expected_sources basename -> benchmark_documents.id | False |
| Q39: 星图推荐引擎是否已经切换到 Chroma？ | distractor | True | D13, D19 | expected_sources basename -> benchmark_documents.id | False |
| Q40: 流光监控 Atlas 事件服务，是否意味着流光在业务上依赖 Atlas？ | distractor | True | D16 | expected_sources basename -> benchmark_documents.id | False |
| Q41: 北极星为天枢提供检索索引，是否意味着北极星依赖天枢？ | distractor | True | D10 | expected_sources basename -> benchmark_documents.id | False |
| Q42: Atlas 事件服务的负责人是陈航吗？ | distractor | True | D08 | expected_sources basename -> benchmark_documents.id | False |
| Q43: 曙光客户门户和星图推荐引擎是同一个项目吗？ | distractor | True | D12, D04 | expected_sources basename -> benchmark_documents.id | False |
| Q44: 河图数据湖当前使用 Elasticsearch 吗？ | distractor | True | D14 | expected_sources basename -> benchmark_documents.id | False |
| Q45: 猎户座权限服务是否由赵启负责？ | distractor | True | D11 | expected_sources basename -> benchmark_documents.id | False |
| Q46: 华远能源是否直接调用北极星检索平台？ | distractor | True | D20, D12 | expected_sources basename -> benchmark_documents.id | False |
| Q47: 云桥 API 网关和 Atlas 事件服务承担的是同一种职责吗？ | distractor | True | D17, D08 | expected_sources basename -> benchmark_documents.id | False |
| Q48: 晨星报表系统和星图推荐引擎是否都使用 PostgreSQL 作为检索存储？ | distractor | True | D15, D13 | expected_sources basename -> benchmark_documents.id | False |
| Q49: 星河智联科技集团的 CEO 是谁？ | abstention | False | — | requires_abstention -> no relevant document | False |
| Q50: 星河智联科技集团的 CFO 是谁？ | abstention | False | — | requires_abstention -> no relevant document | False |
| Q51: 人力资源部负责人是谁？ | abstention | False | — | requires_abstention -> no relevant document | False |
| Q52: 南斗项目的负责人是谁？ | abstention | False | — | requires_abstention -> no relevant document | False |
| Q53: 天枢知识平台生产环境使用的 Kubernetes 版本是多少？ | abstention | False | — | requires_abstention -> no relevant document | False |
| Q54: 北极星检索平台的日活用户数是多少？ | abstention | False | — | requires_abstention -> no relevant document | False |
| Q55: 星河智联科技集团 2026 年第二季度营收是多少？ | abstention | False | — | requires_abstention -> no relevant document | False |
| Q56: 华远能源的本次部署位于哪个城市？ | abstention | False | D20 | expected_sources basename -> benchmark_documents.id | False |
| Q57: 猎户座权限服务使用什么具体密码加密算法？ | abstention | False | — | requires_abstention -> no relevant document | False |
| Q58: Atlas 事件服务的 SLA 可用性百分比是多少？ | abstention | False | — | requires_abstention -> no relevant document | False |
| Q59: 北极星检索平台正式上线的具体日期是什么？ | abstention | False | — | requires_abstention -> no relevant document | False |
| Q60: 陈航的手机号码是多少？ | abstention | False | — | requires_abstention -> no relevant document | False |
