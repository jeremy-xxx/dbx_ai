# 阶段 7c：手写 RAG —— 切块、向量化、检索、生成、评估

> **学习目标**
> 1. 理解 RAG 解决什么问题，以及它的完整流程；
> 2. 亲手实现每一步：切块 → 向量化 → 存储 → 检索 → 拼提示词生成 → 评估；
> 3. 会用 MLflow 评估 RAG：**检索准不准、回答有没有依据、答得对不对**；
> 4. 了解 Databricks 的托管向量检索服务 **AI Search**（原名 Vector Search）；
> 5. 能说清三种“让大模型用上公司资料”的方式各自适合什么场景。
>
> **耗时**：1.5–2 小时 **前置**：阶段 7（至少运行过 `07a`） **费用**：embedding 和大模型按 token 计费，很少；可选的 AI Search 端点会持续计费

---

## 7c.1 为什么需要 RAG

大模型的知识来自训练数据，**不知道你们公司的内部资料**：维保规程、质保政策、最新的 FAQ。让它用上这些资料，有三种做法：

| 做法 | 怎么做 | 问题 |
|---|---|---|
| 微调模型 | 用公司资料继续训练模型 | 贵、慢；资料一改就要重新训练；难以说明答案出自哪里 |
| **全部塞进提示词** | 每次把所有资料放进提示词 | 资料一多就太长、太贵，甚至超过模型的上下文上限；关键信息容易被淹没 |
| **RAG（检索增强生成）** | 先检索出和问题**最相关的几段**，只把这几段放进提示词 | 要多做“切块、向量化、检索”几步，检索不准时答案也会错 |

RAG 是目前最常用的方案：资料更新后只需重新向量化，不用重训模型；每个回答都能标出出处，方便核对。

## 7c.2 完整流程

```
离线准备（资料变化时才重跑）
  规程文档 ──① 切块──► ② 向量化 ──► ③ 存储（Delta 表 / 向量索引）

在线回答（每个问题都跑）
  问题 ──② 向量化──► ④ 检索最相似的 K 块 ──► ⑤ 拼进提示词 ──► 大模型回答（带引用编号）

质量评估
  ⑥ 检索准不准 / 回答有没有依据 / 答得对不对
```

本项目里各部分的对应关系：

| 步骤 | `07c_rag.py` 中的位置 | 用到的 Databricks 能力 |
|---|---|---|
| ① 切块 | 第 1 节 | 读取 `07a` 写入的 `policy_texts` 表 |
| ② 向量化 | 第 2 节 | Unity Gateway 上的 embedding 模型 `system.ai.qwen3-embedding-0-6b` |
| ③ 存储 | 第 3 节；可选第 8 节 | Delta 表 `policy_chunks`；可选 AI Search 索引 |
| ④ 检索 | 第 4 节 | numpy 计算余弦相似度；MLflow Tracing 标记为 RETRIEVER |
| ⑤ 生成 | 第 5 节 | Unity Gateway 上的聊天模型（`LLM_MODEL`） |
| ⑥ 评估 | 第 7 节 | `mlflow.genai.evaluate` + RAG 专用评分器 |

## 7c.3 切块（Chunking）

**为什么要切块？**
- 一篇文档里通常只有一小段和问题相关，整篇放进去既浪费 token，又会稀释重点；
- 一个向量的表达能力有限，文字越长，越难同时表达其中的所有意思。

**常见的切块方式**

| 方式 | 说明 | 适合 |
|---|---|---|
| 固定长度 | 每 N 个字切一刀，相邻块重叠一部分 | 没有结构的长文本 |
| **按文档结构** | 按标题、段落切 | Markdown、Word 这类有章节的文档（**本项目用的方式**） |
| 按语义 | 用模型判断话题在哪里转换 | 结构混乱的长文本 |

本项目的规则：按 `## ` 二级标题切成章节；章节超过 `MAX_CHARS`（350 字）再按段落切；FAQ 每个“问 + 答”单独成块。3 篇文档切成约 18 块。

**一个提升效果的小技巧**：向量化时，在每块前面加上“文档名 + 章节名”，比如《动力电池质保政策》5. 审批。像“5. 审批”这种只有两行的短块，单看内容很难判断它在讲什么，加上标题后检索准确得多（练习 4 会验证这一点）。

## 7c.4 向量化（Embedding）

Embedding 模型把一段文字变成一串数字，也就是**向量**（本项目约 1000 维）。它的关键特性是：**意思相近的文字，向量也相近**。

```
“电芯压差多大就危险了？”        → [0.12, -0.03, 0.88, ...]  ┐
“电芯最大压差 ≥ 150 mV 为安全红线” → [0.10, -0.05, 0.85, ...]  ┘ 很接近（意思相关）
“冬天续航为什么下降？”           → [-0.40, 0.67, 0.02, ...]    离得远（意思无关）
```

**选模型要看语言**：规程是中文，所以用支持多语言的 `qwen3-embedding-0-6b`。你的 workspace 里还能看到 `gte-large-en`（只懂英文）和 `bge_m3`（多语言）。

**问题和资料必须用同一个 embedding 模型**：不同模型的向量不在同一个“坐标系”里，混用没法比较。

调用方式和聊天模型一样，都走 Unity Gateway：
```python
client = get_llm_client()
resp = client.embeddings.create(model=EMBEDDING_MODEL, input=["文本1", "文本2"])
vectors = [item.embedding for item in resp.data]
```

## 7c.5 检索与生成

**检索**：把问题向量化，和每一块算**余弦相似度**（两个向量方向越一致，值越接近 1），取分数最高的 K 块（本项目 `TOP_K = 3`）。

这是**语义检索**，比较的是意思而不是字面。notebook 第 4 节的例子：问“电芯压差多大就危险了？”，资料原文是“电芯最大压差 ≥ 150 mV 为安全红线”，问题里没有“红线”“150”这些词，依然能检索到。用传统的关键词搜索就找不到。

**生成**：把检索到的块编上号放进提示词。提示词有三个关键要求：
1. **只根据资料回答**，减少模型自由发挥；
2. **标注引用编号** `[1]`、`[2]`，让人能核对出处；
3. **资料里没有就说没有**，这是防止编造的最后一道防线。

```
【参考资料】
[1] 《动力电池维保规程》3. 安全红线
满足以下任一条件……电芯最大压差 ≥ 150 mV……
[2] ……

【问题】电芯压差多大就危险了？发现后要怎么处理？
```

## 7c.6 评估 RAG

RAG 出错有两类原因：**检索错了**（没找到相关资料），或**生成错了**（找到了但答错、编造）。评估要能区分这两种：

| 评分器 | 检查什么 | 出问题说明 |
|---|---|---|
| `RetrievalRelevance` | 检索到的每一块是否和问题相关 | **检索**有问题：调切块、换 embedding 模型、调 TOP_K |
| `RetrievalGroundedness` | 回答是否都有资料依据 | **生成**有问题：模型在编造，加强提示词约束 |
| `Correctness` | 回答是否包含标准答案要点（`expected_facts`） | 端到端结果不对，再结合上面两项定位原因 |
| `Guidelines`（cites_sources） | 是否标注了引用编号 | 提示词没被遵守 |

检索类评分器靠 Trace 里标记为 **RETRIEVER** 的步骤找到检索结果，所以 `retrieve` 函数加了 `@mlflow.trace(span_type=SpanType.RETRIEVER)`，返回值是 MLflow 的 `Document` 列表。

**经验**：RAG 效果不好时，**先看检索，再看生成**。大多数问题出在检索：点开答错那条 Trace 的 `retrieve` 步骤，看检索到的是哪几块。

notebook 第 7 节还用同一个评估集评估了**全量资料**的做法（对照组），对比两者的 `correctness`。

## 7c.7 接入 Agent（练习）

阶段 7 代码版 Agent 的 `get_policy_doc` 工具是按名字取回整篇文档。换成 RAG 后，Agent 拿到的是和问题最相关的几段。有两种接入方式：

**方式一：Playground，零代码**（需要先完成第 8 节，建好 AI Search 索引）
在 Playground 选择 Qwen 模型 → **Tools → Vector Search** 标签 → 选择 `policy_chunks_index` → 提问“电芯压差多少算红线？”，观察模型怎么调用检索工具。

**方式二：修改 `agent.py`**（思路）
1. 在 `TOOLS` 里新增工具 `search_policy(question)`，描述写清楚“根据问题检索规程中最相关的段落”；
2. 在 `call_tool` 里实现：用 `EMBEDDING_MODEL` 把问题向量化 → 调用 AI Search 索引的 `similarity_search` → 把结果拼成带编号的文本返回；
3. System Prompt 里把“调用 `get_policy_doc` 阅读规程”改成“调用 `search_policy` 检索相关规程”；
4. 部署时在 `resources` 里声明索引：`DatabricksVectorSearchIndex(index_name=AI_SEARCH_INDEX)`；
5. 重跑 `07b` 的评估，和原来的版本对比得分。

## 7c.8 AI Search（可选，notebook 第 8 节）

**AI Search**（原名 Vector Search）是 Databricks 托管的向量检索服务。第 4 节用 numpy 把问题和每一块都算一遍，十几块没问题；资料有几万、几百万块时就需要专门的索引来加速。

```
policy_chunks（Delta 表）──自动同步──► policy_chunks_index（AI Search 索引）◄── AI Search 端点（计算资源）
                                                     ▲
                                  问题向量 ──similarity_search──┘
```

| 概念 | 说明 |
|---|---|
| **端点（endpoint）** | 运行索引的计算资源，常驻、持续计费，一个端点可以放多个索引 |
| **Delta Sync 索引** | 以 Delta 表为数据源，表变化后同步到索引（`TRIGGERED` 手动触发，`CONTINUOUS` 持续同步） |
| **自己算向量 vs 托管向量** | 本项目自己算好向量存在 `embedding` 列（`embedding_vector_column`）；也可以只给文本列，让 AI Search 调用 embedding 模型自动算 |

Python 包也随产品改了名：新包是 `databricks-ai-search`（`from databricks.ai_search.client import AISearchClient`），旧包 `databricks-vectorsearch` 仍可用，但会提示已弃用。

## 7c.9 三种方式对比

| | 全量资料（阶段 7 的 `get_policy_doc`） | Agent Bricks Knowledge Assistant（阶段 7.2） | 手写 RAG（本阶段） |
|---|---|---|---|
| 实现成本 | 最低 | 低：点几下按钮 | 高：每一步自己写 |
| 能处理的资料量 | 很少（几篇短文档） | 大 | 大（配合 AI Search） |
| 可控性 | — | 低：切块、检索方式由平台决定 | 高：每一步都能调 |
| 能否看到内部过程 | — | 部分 | 完全可见（Trace） |
| 适合 | 资料极少的原型 | 标准文档问答，快速上线 | 需要定制、要嵌入自己的 Agent、要深度调优 |

## 7c.10 练习
1. **调切块大小**：`MAX_CHARS` 改成 120 或 800，对比评估结果；
2. **调 TOP_K**：改成 1 和 6，看 `retrieval_relevance` 和 `correctness` 怎么变；
3. **换向量模型**：`EMBEDDING_MODEL` 换成只懂英文的 `system.ai.gte-large-en`，看中文检索变差多少；
4. **去掉“文档名 + 章节名”前缀**：只用 `content` 向量化，看“审批”“记录要求”这类短块还能不能被检索到；
5. **接入 Agent**：按 7c.7 的方式一或方式二；
6. **思考题**：用户问“商用车质保期限”，资料原文是“commercial_van”。语义检索能找到吗？如果找不到，有什么办法？（提示：混合检索，即语义检索 + 关键词检索；或在切块时补充同义词）

## 7c.11 检查点
- [ ] `policy_chunks` 表有约 18 行，每行有 `embedding` 列
- [ ] 第 4 节的检索结果里，“压差”问题排第一的是“安全红线”那一块
- [ ] 打开一条 `rag_answer` 的 Trace，能看到 retrieve → embeddings → chat.completions 三步，以及检索到的内容
- [ ] 评估结果里有 `retrieval_relevance`、`retrieval_groundedness`、`correctness`，并能解释 RAG 和全量资料的差异
- [ ] “年终奖”问题，模型回答了“资料中没有相关信息”

## 7c.12 常见坑

| 现象 | 原因 / 解决 |
|---|---|
| `embeddings.create` 报 404 / does not exist | 左侧 **AI Gateway → Models** 确认 embedding 模型名；可换 `system.ai.gte-large-en`（英文）或其它可用的多语言模型 |
| `policy_texts` 表不存在 | 先运行 `07a_agent_tools` |
| 检索结果明显不相关 | 检查问题和资料是否用了**同一个** embedding 模型；中文资料不要用只懂英文的模型 |
| 检索类评分器结果为空 | `retrieve` 函数需要 `@mlflow.trace(span_type=SpanType.RETRIEVER)`，并返回 `Document` 列表 |
| `DeprecationWarning: databricks-vectorsearch is deprecated` | 包已改名为 `databricks-ai-search`，按提示换包和 import 即可 |
| 建 AI Search 索引报 embedding 列类型错误 | 向量列需要 `array<float>`，第 3 节已转换 |
| AI Search 端点创建失败 | Free Edition 可能不支持或有数量限制；跳过第 8 节不影响前面的学习 |

➡️ 下一步：[阶段 8：综合项目](08-capstone.md)
