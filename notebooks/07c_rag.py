# Databricks notebook source
# MAGIC %md
# MAGIC # 07c · 手写 RAG：切块 → 向量化 → 检索 → 生成 → 评估
# MAGIC
# MAGIC 对应文档：`docs/07c-rag.md`
# MAGIC
# MAGIC **RAG（Retrieval-Augmented Generation，检索增强生成）**：大模型不知道公司内部资料，
# MAGIC 那就先从资料里**检索**出和问题最相关的几段，再把它们放进提示词，让模型**基于资料回答**。
# MAGIC
# MAGIC ```
# MAGIC 离线准备（本 notebook 第 1–3 节，跑一次）：
# MAGIC   规程文档 ──① 切块──► ② 向量化（embedding）──► ③ 存成 Delta 表（可选：建 AI Search 索引）
# MAGIC
# MAGIC 在线回答（第 4–6 节，每个问题都跑）：
# MAGIC   问题 ──② 向量化──► ④ 检索最相似的 K 块 ──► ⑤ 拼进提示词 ──► 大模型回答（带引用编号）
# MAGIC
# MAGIC 质量评估（第 7 节）：
# MAGIC   ⑥ 检索得准不准？回答有没有依据？答得对不对？
# MAGIC ```
# MAGIC
# MAGIC 前置：运行过 `07a_agent_tools`（它把规程文档写进了 `policy_texts` 表）。

# COMMAND ----------

# MAGIC %pip install -U -q "mlflow[databricks]>=3.1" databricks-sdk openai databricks-ai-search

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import Document, SpanType

# 👀 界面：本 notebook 的 Trace 和评估结果记录到 Experiments → ev-battery-genai
mlflow.set_experiment(GENAI_EXPERIMENT_PATH)
mlflow.openai.autolog()   # 自动追踪 embedding 和 chat 调用

client = get_llm_client()   # 同一个客户端既能调聊天模型，也能调 embedding 模型（都走 Unity Gateway）
print("聊天模型:", LLM_MODEL, "| 向量模型:", EMBEDDING_MODEL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 切块（Chunking）
# MAGIC 为什么要切块？
# MAGIC - 检索的粒度：一整篇文档里只有一小段和问题相关，整篇塞给模型既浪费 token，又会稀释重点；
# MAGIC - 向量的表达能力有限：一段文字越长，一个向量越难同时表达其中的所有意思。
# MAGIC
# MAGIC 本节的切块规则（按文档结构切，是最常用也最稳妥的做法）：
# MAGIC 1. 按 Markdown 二级标题 `## ` 切成章节；
# MAGIC 2. 章节太长（超过 `MAX_CHARS`）时，再按空行（段落）切开；
# MAGIC 3. FAQ 里每个 “问 + 答” 单独成一块。

# COMMAND ----------

import re

MAX_CHARS = 350   # 每块最多约 350 个字符

policy_rows = spark.table(POLICY_TABLE).select("doc_name", "content").collect()
assert policy_rows, "policy_texts 表为空，请先运行 07a_agent_tools"

chunk_rows = []
for row in policy_rows:
    doc_title = row.content.split("\n", 1)[0].lstrip("# ").strip()   # 第一行是文档标题
    for section_text in re.split(r"\n(?=## )", row.content):           # 按 “## ” 切成章节
        lines = section_text.strip().split("\n")
        if lines[0].startswith("## "):
            section = lines[0][3:].strip()
            body_lines = lines[1:]
        else:
            section = "概述"
            body_lines = [line for line in lines if not line.startswith("# ")]
        body_lines = [line for line in body_lines if not line.startswith("> ")]   # 去掉“虚构数据”提示行
        paragraphs = [p.strip() for p in "\n".join(body_lines).split("\n\n") if p.strip()]

        buffer = ""
        for p in paragraphs:
            if p.startswith("**Q"):   # FAQ：一问一答单独成块，章节名用问题本身
                if buffer:
                    chunk_rows.append((row.doc_name, doc_title, section, buffer))
                    buffer = ""
                chunk_rows.append((row.doc_name, doc_title, p.split("\n", 1)[0].strip("* "), p))
            elif buffer and len(buffer) + len(p) > MAX_CHARS:
                chunk_rows.append((row.doc_name, doc_title, section, buffer))
                buffer = p
            else:
                buffer = (buffer + "\n\n" + p).strip()
        if buffer:
            chunk_rows.append((row.doc_name, doc_title, section, buffer))

chunks = pd.DataFrame(chunk_rows, columns=["doc_name", "doc_title", "section", "content"])
chunks["chunk_id"] = chunks.doc_name + "-" + chunks.groupby("doc_name").cumcount().astype(str).str.zfill(2)
# 给每块加上“文档名 + 章节名”再去向量化：让 “3. 安全红线” 这种短块也带上上下文，检索更准
chunks["text_for_embedding"] = "《" + chunks.doc_title + "》" + chunks.section + "\n" + chunks.content
chunks["n_chars"] = chunks.content.str.len()

print(f"{len(policy_rows)} 篇文档 → {len(chunks)} 块")
display(chunks[["chunk_id", "section", "n_chars", "content"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 向量化（Embedding）
# MAGIC Embedding 模型把一段文字变成一串数字（向量）。**意思相近的文字，向量也相近**。
# MAGIC
# MAGIC 选模型要看语言：规程是中文，所以用支持中文的 `qwen3-embedding-0-6b`。
# MAGIC 如果换成只懂英文的 `gte-large-en`，检索效果会明显变差（练习 3）。

# COMMAND ----------

texts = chunks.text_for_embedding.tolist()
vectors = []
for start in range(0, len(texts), 16):   # 分批调用，每批 16 块
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts[start:start + 16])
    vectors += [item.embedding for item in resp.data]

chunks["embedding"] = vectors
DIM = len(vectors[0])
print(f"每块变成一个 {DIM} 维向量。第一块的前 5 个数：", np.round(vectors[0][:5], 4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 存成 Delta 表
# MAGIC 切块和向量是“离线准备”的结果，存下来后每次回答问题都直接用，不用重算。
# MAGIC 开启 Change Data Feed（变更数据流）是为了可选的第 8 节：AI Search 索引靠它自动同步表的变化。

# COMMAND ----------

from pyspark.sql import functions as F

# 👀 界面：Catalog → main → ev_battery_lab → Tables 出现 policy_chunks（每块一行，含 embedding 列）
(spark.createDataFrame(chunks.drop(columns=["n_chars"]))
    .withColumn("embedding", F.col("embedding").cast("array<float>"))   # AI Search 要求 array<float>
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(CHUNK_TABLE))
spark.sql(f"ALTER TABLE {CHUNK_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"✅ 写入 {CHUNK_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 检索：找出和问题最相似的几块
# MAGIC 问题也用**同一个** embedding 模型变成向量，然后和每一块算**余弦相似度**（两个向量的夹角，越接近 1 越相似）。
# MAGIC
# MAGIC 本项目只有十几块，直接用 numpy 全部算一遍就行。资料有成千上万块时，就需要第 8 节的向量索引来加速。
# MAGIC
# MAGIC `@mlflow.trace(span_type=SpanType.RETRIEVER)` 把这一步标记为“检索”步骤：
# MAGIC Trace 里能看到检索到了哪几块；第 7 节的检索类评分器也靠这个标记找到检索结果。

# COMMAND ----------

chunk_table = spark.table(CHUNK_TABLE).toPandas()
chunk_matrix = np.array(chunk_table.embedding.tolist())
chunk_matrix = chunk_matrix / np.linalg.norm(chunk_matrix, axis=1, keepdims=True)   # 先归一化，点积 = 余弦相似度
TOP_K = 3

@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve(question: str, k: int = TOP_K) -> list[Document]:
    q = np.array(client.embeddings.create(model=EMBEDDING_MODEL, input=[question]).data[0].embedding)
    q = q / np.linalg.norm(q)
    scores = chunk_matrix @ q                    # 和每一块的余弦相似度
    top = np.argsort(-scores)[:k]                # 分数最高的 k 块
    return [
        Document(
            id=chunk_table.chunk_id[i],
            page_content=chunk_table.content[i],
            metadata={"doc_uri": chunk_table.doc_name[i], "doc_title": chunk_table.doc_title[i],
                      "section": chunk_table.section[i], "score": round(float(scores[i]), 4)},
        )
        for i in top
    ]

# 👀 界面：每次检索都会在 Experiments → ev-battery-genai → Traces 留下一条 RETRIEVER 类型的记录
for question in ["电芯压差多大就危险了？", "LFP 电池能不能天天充满？"]:
    print(f"❓ {question}")
    for d in retrieve(question):
        print(f"   {d.metadata['score']:.3f}  [{d.id}] {d.metadata['section']}")

# COMMAND ----------

# MAGIC %md
# MAGIC 注意第一个问题：资料里写的是“电芯最大压差 ≥ 150 mV 为安全红线”，问题里根本没有“红线”“150”这些词，
# MAGIC 但依然能检索到。这就是**语义检索**和**关键词搜索**的区别：比较的是意思，不是字面。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 生成：把检索结果拼进提示词
# MAGIC 提示词的三个关键要求：
# MAGIC 1. **只根据资料回答**：减少模型“自由发挥”；
# MAGIC 2. **标注引用编号** `[1]`：让人能核对答案出自哪段资料；
# MAGIC 3. **资料里没有就说没有**：这是防止编造的最后一道防线。

# COMMAND ----------

RAG_SYSTEM_PROMPT = """你是电动车公司的电池售后知识助手，只能根据【参考资料】回答问题：
1. 回答简洁，在用到资料的句子后用 [1]、[2] 标注资料编号；
2. 资料中没有答案时，直接回答“资料中没有相关信息”，不要编造；
3. 使用简体中文。"""

@mlflow.trace
def rag_answer(question: str) -> str:
    docs = retrieve(question)
    context = "\n\n".join(
        f"[{i}] 《{d.metadata['doc_title']}》{d.metadata['section']}\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"【参考资料】\n{context}\n\n【问题】{question}"},
        ],
    )
    return resp.choices[0].message.content

# 👀 界面：每次调用在 Traces 里是一棵树：rag_answer → retrieve（检索）→ embeddings → chat.completions（生成）
for question in ["电芯压差多大就危险了？发现后要怎么处理？", "公司的年终奖是怎么发放的？"]:
    print(f"❓ {question}\n💬 {rag_answer(question)}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 对照组：不检索，把全部资料塞进提示词
# MAGIC 这就是阶段 7 代码版 Agent 里 `get_policy_doc` 的做法（按名字取整篇文档）。
# MAGIC 本项目资料很少，这样也能答对；但资料一多就不行了：提示词太长、太贵，甚至超过模型的上下文长度上限。

# COMMAND ----------

all_docs = "\n\n".join(row.content for row in policy_rows)

@mlflow.trace
def full_context_answer(question: str) -> str:
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"【参考资料】\n{all_docs}\n\n【问题】{question}"},
        ],
    )
    return resp.choices[0].message.content

rag_context_chars = chunks.n_chars.sort_values(ascending=False).head(TOP_K).sum()   # RAG 提示词资料部分的上限
print(f"全部资料：{len(all_docs)} 字；RAG 每次最多只放 {TOP_K} 块，约 {rag_context_chars} 字"
      f"（只有约 {rag_context_chars / len(all_docs):.0%}）")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. 评估：检索准不准、回答有没有依据、答得对不对
# MAGIC
# MAGIC | 评分器 | 检查什么 | 需要什么 |
# MAGIC |---|---|---|
# MAGIC | `RetrievalRelevance` | 检索到的每一块是否和问题相关 | Trace 里的 RETRIEVER 步骤 |
# MAGIC | `RetrievalGroundedness` | 回答是否**有资料依据**（没有编造） | Trace 里的 RETRIEVER 步骤 |
# MAGIC | `Correctness` | 回答是否包含标准答案里的要点 | 评估集里的 `expected_facts` |
# MAGIC | `Guidelines` | 自定义规则：是否标注了引用编号 | 规则描述 |
# MAGIC
# MAGIC 评估集里特意放了一个资料中没有答案的问题（年终奖），检查模型会不会编造。

# COMMAND ----------

from mlflow.genai.scorers import Correctness, Guidelines, RetrievalGroundedness, RetrievalRelevance

eval_data = [
    {"inputs": {"question": "电芯压差多少算安全红线？触发后要怎么处理？"},
     "expectations": {"expected_facts": ["电芯最大压差达到或超过 150 mV 属于安全红线", "要立即停止使用快充", "要在 24 小时内进站检测"]}},
    {"inputs": {"question": "商用物流车的电池质保是几年、多少公里？"},
     "expectations": {"expected_facts": ["质保 5 年", "质保 300,000 公里", "以先到者为准"]}},
    {"inputs": {"question": "车在质保期内，SOH 是 75%，可以免费换电池吗？"},
     "expectations": {"expected_facts": ["SOH 低于 70% 才能免费更换", "75% 属于正常衰减，不能免费更换", "电池检测是免费的"]}},
    {"inputs": {"question": "磷酸铁锂电池可以每天充到 100% 吗？"},
     "expectations": {"expected_facts": ["可以每天充到 100%", "建议每周至少充满一次以校准电量显示"]}},
    {"inputs": {"question": "电池被水泡了，在质保期内能免费修吗？"},
     "expectations": {"expected_facts": ["涉水、泡水导致的损坏不在质保范围内"]}},
    {"inputs": {"question": "公司的年终奖是怎么发放的？"},
     "expectations": {"expected_facts": ["说明资料中没有相关信息"]}},
]

cites_sources = Guidelines(
    name="cites_sources",
    guidelines="用到资料时必须用 [1]、[2] 这样的编号标注来源；如果回答是“资料中没有相关信息”，视为通过。",
)

# 👀 界面：Experiments → ev-battery-genai → Evaluations（或 Evaluation runs）出现两次评估，可逐条查看得分和裁判理由
rag_results = mlflow.genai.evaluate(
    data=eval_data, predict_fn=rag_answer,
    scorers=[RetrievalRelevance(), RetrievalGroundedness(), Correctness(), cites_sources],
)
full_results = mlflow.genai.evaluate(
    data=eval_data, predict_fn=full_context_answer,
    scorers=[Correctness(), cites_sources],   # 没有检索步骤，检索类评分器不适用
)

comparison = pd.DataFrame({"RAG": pd.Series(rag_results.metrics), "全量资料": pd.Series(full_results.metrics)})
display(comparison.reset_index().rename(columns={"index": "指标"}))

# COMMAND ----------

# MAGIC %md
# MAGIC **怎么看结果**：
# MAGIC - 两者的 `correctness` 应该接近：资料只有 3 篇，全塞进去也能答对；
# MAGIC - RAG 的提示词短得多（见第 6 节），资料越多，这个优势越大；
# MAGIC - 如果 RAG 的某题答错，先看 `retrieval_relevance`：**大多数 RAG 问题出在检索，不在生成**。
# MAGIC   点开那条 Trace 的 retrieve 步骤，看检索到的是哪几块。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.（可选）建 AI Search 向量索引
# MAGIC **AI Search**（原名 Vector Search）是 Databricks 的托管向量检索服务。资料多时，它负责快速找到最相似的块，替代第 4 节的 numpy 全量计算。
# MAGIC
# MAGIC ⚠️ 需要创建一个 **AI Search 端点**（常驻计算资源，会持续计费；Free Edition 可能有数量限制或不可用）。
# MAGIC 学完记得在 `99_cleanup` 中删除。把 `RUN_AI_SEARCH` 改成 True 再运行。

# COMMAND ----------

RUN_AI_SEARCH = False

if RUN_AI_SEARCH:
    from databricks.ai_search.client import AISearchClient

    search_client = AISearchClient(disable_notice=True)

    # 👀 界面：左侧 Compute → AI Search（或 Vector Search）标签页出现这个端点（首次创建约 10–20 分钟）
    if not search_client.endpoint_exists(AI_SEARCH_ENDPOINT):
        search_client.create_endpoint_and_wait(name=AI_SEARCH_ENDPOINT, endpoint_type="STANDARD")

    if search_client.index_exists(endpoint_name=AI_SEARCH_ENDPOINT, index_name=AI_SEARCH_INDEX):
        index = search_client.get_index(endpoint_name=AI_SEARCH_ENDPOINT, index_name=AI_SEARCH_INDEX)
        index.sync()   # 表重写过，触发一次同步
    else:
        # 👀 界面：Catalog → main → ev_battery_lab 下出现 policy_chunks_index（索引也是 UC 对象）
        search_client.create_delta_sync_index_and_wait(
            endpoint_name=AI_SEARCH_ENDPOINT,
            index_name=AI_SEARCH_INDEX,
            source_table_name=CHUNK_TABLE,       # 索引的数据来源：第 3 节的 Delta 表
            primary_key="chunk_id",
            pipeline_type="TRIGGERED",           # 手动触发同步（另一种 CONTINUOUS 会持续同步，更贵）
            embedding_dimension=DIM,
            embedding_vector_column="embedding", # 向量已经自己算好了（self-managed embeddings）
        )
        index = search_client.get_index(endpoint_name=AI_SEARCH_ENDPOINT, index_name=AI_SEARCH_INDEX)

    question = "电芯压差多大就危险了？"
    q = client.embeddings.create(model=EMBEDDING_MODEL, input=[question]).data[0].embedding
    result = index.similarity_search(query_vector=q, columns=["chunk_id", "section"], num_results=TOP_K)
    print(f"❓ {question}（AI Search 检索结果，最后一列是相似度分数）")
    for r in result["result"]["data_array"]:
        print("  ", r)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✍️ 练习
# MAGIC 1. **调切块大小**：把 `MAX_CHARS` 改成 120 或 800，重跑第 1–7 节，对比评估结果。块太小会丢上下文，太大会混入无关内容。
# MAGIC 2. **调 TOP_K**：改成 1 和 6，看 `retrieval_relevance` 和 `correctness` 怎么变。
# MAGIC 3. **换向量模型**：把 `00_config` 的 `EMBEDDING_MODEL` 换成只懂英文的 `system.ai.gte-large-en`，看中文检索变差多少。
# MAGIC 4. **去掉“文档名 + 章节名”前缀**：第 1 节只用 `content` 去向量化，看短块（如“审批”“记录要求”）还能不能被检索到。
# MAGIC 5. **接入 Agent**：见文档 7c.7，用检索替换 Agent 里“整篇取回”的 `get_policy_doc`。
