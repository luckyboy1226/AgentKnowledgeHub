# Flood Emergency LongDoc Benchmark v3.0 FINAL

状态：**READY_FOR_PARENT_CHILD_AB**

## 规模
- 20 篇城市洪涝应急长文档
- 每篇 2580–2702 个中文字符，平均 2639.3
- 每篇包含标题、摘要、关键词
- 60 道问题；single_hop / multi_hop / constraint / distractor / abstention 各 12 道
- 48 道可回答 + 12 道拒答
- Gold Evidence：83 条

## Parent-Child 专项结构
按项目当前默认参数：
- Parent target = 1000 estimated tokens
- Parent max = 1400
- Child target = 250
- Child overlap = 50

实际验证：
- 每篇正好 3 个 Parent
- 每篇 17–18 个 Child
- child_sufficient = 15
- same_parent_multi_child = 15
- cross_parent = 6
- cross_document = 12
- abstention = 12
- Gold Evidence Parent 分布：{0: 52, 1: 24, 2: 7}

## 冻结规则
这版之后不再修改主体数据。下一步只做真实入库、运行时 Child/Parent ID 映射和 paired A/B。

A 组：Child Only  
B 组：同一 Child 检索结果 + Parent Expansion

两组固定 Query、候选池、排序、Top-K 和 Token Budget，仅改变 Parent Expansion。

## 数据性质
该数据集为受控合成数据，不得描述为真实政府内部材料或真实灾情数据。
