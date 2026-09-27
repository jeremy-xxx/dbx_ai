# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · 把 Playground 里的操作变成代码
# MAGIC
# MAGIC 对应文档：`docs/01-playground.md`

# COMMAND ----------

# MAGIC %pip install -U -q "mlflow[databricks]>=3.1" databricks-sdk openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 找到可用的基础模型
# MAGIC 2026-09 起，基础模型统一由 **Unity Gateway** 提供，模型注册在 Unity Catalog 的 `system.ai` 下：
# MAGIC - 在左侧 **Unity Gateway → Models** 中浏览可调用的模型（名称形如 `system.ai.qwen3-next-80b-a3b-instruct`），点开模型的 **Metrics** 标签可看到调用记录；
# MAGIC - 旧版 `databricks-*` 端点仍可通过 Serving 页面顶部的 “legacy experience” 访问，对应 `00_config` 中 `LLM_MODE = "legacy"`。
# MAGIC
# MAGIC 下面发一个最小请求，验证 `00_config` 里配置的模型可用。

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
print("LLM_MODE:", LLM_MODE, "| model:", LLM_ENDPOINT)
if LLM_MODE == "legacy":
    print("可用的旧版基础模型端点:",
          sorted(e.name for e in w.serving_endpoints.list() if e.name.startswith("databricks-")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. OpenAI 兼容客户端
# MAGIC `get_llm_client()`（定义在 `00_config`）返回一个配置好 `base_url` 和鉴权的 `openai.OpenAI` 对象：
# MAGIC - gateway 模式：`https://<workspace>/ai-gateway/mlflow/v1`
# MAGIC - legacy 模式：`https://<workspace>/serving-endpoints`

# COMMAND ----------

client = get_llm_client()

resp = client.chat.completions.create(
    model=LLM_ENDPOINT,
    messages=[{"role": "user", "content": "用通俗的语言解释什么是动力电池的 SOH（健康度），以及 LFP 和 NMC 电池在寿命上的区别。"}],
    max_tokens=400,
)
print(resp.choices[0].message.content)
print("\nusage:", resp.usage)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. System Prompt + Temperature 实验
# MAGIC 同一个问题问 3 次：temperature=0 时回答几乎一致，temperature 高时更发散。
# MAGIC 售后场景需要“口径一致”，通常选低 temperature。

# COMMAND ----------

SYSTEM_PROMPT = """你是一家电动车公司的资深电池售后工程师。回答必须：
1) 使用简体中文；2) 不超过 150 字；3) 涉及安全风险时先给出安全提示；4) 最后给出一条可执行的建议。"""

def ask(question: str, temperature: float, system: str = SYSTEM_PROMPT) -> str:
    r = client.chat.completions.create(
        model=LLM_ENDPOINT,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
        temperature=temperature,
        max_tokens=300,
    )
    return r.choices[0].message.content

q = "车主说最近快充时电池包附近很烫，仪表还亮了黄灯，怎么回复？"
for t in [0.0, 1.2]:
    print(f"========== temperature={t} ==========")
    for i in range(3):
        print(f"[{i + 1}] {ask(q, t)}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 结构化 JSON 输出：车主反馈分类
# MAGIC 生产里经常需要模型输出可解析的 JSON（比如写进工单系统）。两道保险：
# MAGIC 1. `response_format={"type": "json_object"}`（部分模型支持）
# MAGIC 2. 代码里 `try/except` 兜底

# COMMAND ----------

import json

FEEDBACKS = [
    "最近快充到 60% 以后就特别慢，而且电池包那里有点烫，仪表还亮了个黄灯。",
    "冬天续航从 400 掉到 260 公里，正常吗？",
    "车子用了三年，感觉比刚买的时候少跑了 30 公里。",
    "停车场闻到车底有焦糊味，还有一点烟。",
]

def classify_feedback(text: str) -> dict:
    prompt = (
        "请分析车主反馈，只输出 JSON，格式：\n"
        '{"issue_type": "charging|range|thermal|warning_light|other", '
        '"safety_risk": "none|low|high", "needs_inspection": true|false, "reason": "..."}\n'
        f"反馈：{text}"
    )
    kwargs = dict(model=LLM_ENDPOINT, messages=[{"role": "user", "content": prompt}], temperature=0)
    try:
        r = client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
    except Exception:
        r = client.chat.completions.create(**kwargs)  # 模型不支持 response_format 时回退
    raw = r.choices[0].message.content
    try:
        return json.loads(raw.strip().removeprefix("```json").removesuffix("```"))
    except json.JSONDecodeError:
        return {"parse_error": True, "raw": raw}

for t in FEEDBACKS:
    print(t, "\n  →", classify_feedback(t))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. MLflow Tracing 预告
# MAGIC 开启 autolog 后，每一次 LLM 调用都会自动记录成一条 **Trace**（输入、输出、耗时、token）。
# MAGIC 运行后去左侧 **Experiments → ev-battery-genai → Traces** 标签页查看。阶段 7 会重度使用它来调试 Agent。
# MAGIC （gateway 模式下，同样的调用也会出现在 **Catalog → system → ai → 你的模型 → Metrics** 里，这是 Unity Gateway 的平台级用量记录。）

# COMMAND ----------

import mlflow

mlflow.set_experiment(GENAI_EXPERIMENT_PATH)
mlflow.openai.autolog()

for t in FEEDBACKS:
    classify_feedback(t)

print("✅ 已生成 Trace，去 Experiments 页面看看")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 在 SQL 里调用大模型：`ai_query`
# MAGIC 对整张表逐行调用 LLM —— 数据分析师不写 Python 也能用大模型。

# COMMAND ----------

display(spark.sql(f"""
SELECT vehicle_id, vehicle_model, battery_chemistry, soh_pct, fast_charge_ratio, max_cell_temp_c_30d,
       ai_query(
         '{LLM_ENDPOINT}',
         concat('你是电池售后工程师。用一句中文（不超过40字）点评这块动力电池的状态：',
                to_json(named_struct('model', vehicle_model, 'chemistry', battery_chemistry,
                                     'age_months', vehicle_age_months, 'soh_pct', soh_pct,
                                     'fast_charge_ratio', fast_charge_ratio,
                                     'max_temp_c', max_cell_temp_c_30d,
                                     'imbalance_mv', cell_voltage_imbalance_mv)))
       ) AS llm_comment
FROM {VEHICLES_TABLE}
ORDER BY soh_pct
LIMIT 5
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✍️ 练习
# MAGIC 1. 修改 `SYSTEM_PROMPT`，让模型用“表格”形式给出：现象 / 可能原因 / 建议。
# MAGIC 2. 换 2 个不同的 `LLM_ENDPOINT`，对 `classify_feedback` 各跑 10 次，统计 JSON 解析成功率，
# MAGIC    以及“焦糊味、冒烟”那条是否 **每次** 都被判为 `safety_risk=high`（安全场景不能有漏判）。
# MAGIC 3. 在 Trace 页面比较两个模型的平均延迟和 token 数。
