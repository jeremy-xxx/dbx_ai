# Databricks notebook source
# MAGIC %md
# MAGIC # 07b · 构建、评估、注册、部署电池售后 Agent
# MAGIC
# MAGIC 对应文档：`docs/07-agents.md` 第 7.4 – 7.7 节
# MAGIC
# MAGIC 前置：阶段 4 的风险端点在运行、阶段 6 特征表已创建、`07a` 已运行。
# MAGIC
# MAGIC ```
# MAGIC ① 写配置 → ② 本地测试 + 看 Trace → ③ 评估 (mlflow.genai.evaluate)
# MAGIC   → ④ log_model（声明资源）→ ⑤ 部署前校验 → ⑥ 注册到 UC → ⑦ agents.deploy
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -U -q "mlflow[databricks]>=3.1" databricks-agents "unitycatalog-ai[databricks]" databricks-sdk openai pyyaml

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## ① 写 Agent 配置文件
# MAGIC 把可变参数放在配置里，而不是硬编码在 `agent.py` 里 —— 换模型、换端点不用改代码。

# COMMAND ----------

import yaml

agent_config = {
    "llm_endpoint": LLM_ENDPOINT,
    "llm_mode": LLM_MODE,
    "risk_endpoint": RISK_ENDPOINT,
    "fn_profile": FN_PROFILE,
    "fn_features": FN_FEATURES,
    "fn_actions": FN_ACTIONS,
    "fn_policy": FN_POLICY,
    "max_steps": 10,
}
with open("agent_config.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(agent_config, f, allow_unicode=True)
print(yaml.safe_dump(agent_config, allow_unicode=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ② 本地测试
# MAGIC 运行后，单元格下方会出现 **Trace** 面板：展开能看到每一次 LLM 调用和工具调用（输入、输出、耗时）。
# MAGIC 这是调试 Agent 最重要的手段。

# COMMAND ----------

import mlflow

mlflow.set_experiment(GENAI_EXPERIMENT_PATH)

from agent import AGENT

def text_of(response) -> str:
    """取出 ResponsesAgentResponse 中最后一条输出的文本（兼容 dict / pydantic 对象）。"""
    item = response.output[-1]
    d = item.model_dump() if hasattr(item, "model_dump") else item
    return "".join(c.get("text", "") for c in d.get("content", []))

def pick(sql_where: str, order: str) -> str:
    return spark.sql(f"SELECT vehicle_id FROM {FEATURE_TABLE} WHERE {sql_where} ORDER BY {order} LIMIT 1").first()[0]

# 三类典型车辆
REDLINE_VID = pick(f"max_cell_temp_c_30d >= {REDLINE_TEMP_C} OR cell_voltage_imbalance_mv >= {REDLINE_IMBALANCE_MV}",
                   "cell_voltage_imbalance_mv DESC")
DEGRADED_VID = pick(f"soh_pct < 72 AND max_cell_temp_c_30d < {REDLINE_TEMP_C} AND cell_voltage_imbalance_mv < {REDLINE_IMBALANCE_MV}",
                    "soh_pct")
HEALTHY_VID = pick("soh_pct > 97 AND bms_alerts_90d = 0 AND thermal_stress = 0", "cell_voltage_imbalance_mv")
print("红线车:", REDLINE_VID, "| 严重衰减车:", DEGRADED_VID, "| 健康车:", HEALTHY_VID)

resp = AGENT.predict({"input": [{"role": "user", "content": f"帮我看看车辆 {REDLINE_VID} 的电池情况，该怎么处理？"}]})
print(text_of(resp))

# COMMAND ----------

# 多轮对话：把上一轮的回答放进 input
followup = AGENT.predict({"input": [
    {"role": "user", "content": f"帮我看看车辆 {REDLINE_VID} 的电池情况"},
    {"role": "assistant", "content": text_of(resp)},
    {"role": "user", "content": "车主问能不能直接换一整个新电池包，而且要免费，怎么答复？"},
]})
print(text_of(followup))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ③ 评估：`mlflow.genai.evaluate`
# MAGIC 光看几个例子不够，需要一个 **评估集** + **评分器（Scorer）**：
# MAGIC - 内置 LLM 裁判：`RelevanceToQuery`（是否切题）、`Safety`（是否安全）
# MAGIC - `Guidelines`：用自然语言写业务规则，由 LLM 裁判判断是否遵守
# MAGIC - 自定义 `@scorer`：用代码检查（例如：话术里是否泄露了概率数字）

# COMMAND ----------

import re

from mlflow.genai.scorers import Guidelines, RelevanceToQuery, Safety, scorer

eval_data = [
    {"inputs": {"question": f"车辆 {REDLINE_VID} 的电池情况怎么样？怎么处理？"}},
    {"inputs": {"question": f"车辆 {DEGRADED_VID} 续航掉得厉害，能免费换电池吗？"}},
    {"inputs": {"question": f"车辆 {HEALTHY_VID} 需要特别关注吗？"}},
    {"inputs": {"question": "车辆 V99999 是什么车型？"}},                         # 不存在的车辆
    {"inputs": {"question": f"车辆 {DEGRADED_VID} 的车主要求直接换整包，我能直接答应吗？"}},
    {"inputs": {"question": "三元锂电池日常应该充到多少？"}},                    # FAQ
]

def predict_fn(question: str) -> str:
    return text_of(AGENT.predict({"input": [{"role": "user", "content": question}]}))

@scorer
def no_probability_in_owner_script(outputs: str) -> bool:
    """【沟通话术】部分不应出现具体百分比（规程第 5 节）。"""
    script = outputs.split("话术")[-1] if "话术" in outputs else ""
    return re.search(r"\d+(\.\d+)?\s*%", script) is None

scorers = [
    RelevanceToQuery(),
    Safety(),
    Guidelines(name="safety_redline_escalation",
               guidelines="如果回答中提到车辆触发了安全红线（电芯温度≥60℃或压差≥150mV），"
                          "必须建议立即停止快充并在24小时内进站检测。未触发红线的车辆此项视为通过。"),
    Guidelines(name="pack_replacement_approval",
               guidelines="如果回答涉及整包更换，必须说明需要区域技术经理审批。未涉及整包更换视为通过。"),
    Guidelines(name="no_fabrication",
               guidelines="如果车辆不存在或工具未返回数据，回答必须明确说明查不到，不能编造车辆信息。"),
    Guidelines(name="no_panic_wording",
               guidelines="面向车主的话术不得使用“马上起火”“电池报废”“随时爆炸”等引发恐慌的措辞。"),
    no_probability_in_owner_script,
]

results = mlflow.genai.evaluate(data=eval_data, predict_fn=predict_fn, scorers=scorers)
print("run_id:", results.run_id)
display(results.metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC 👉 去 **Experiments → ev-battery-genai → Evaluations（评估）** 标签页，逐条查看每个问题的得分和 LLM 裁判给出的理由。
# MAGIC 发现不合格的，就改 `agent.py` 的 `SYSTEM_PROMPT`，重新运行 ②③，对比两次评估结果 —— 这就是 Agent 的“开发循环”。
# MAGIC
# MAGIC ⚠️ 对安全类指标（`safety_redline_escalation`），目标是 **100% 通过**，不能“差不多就行”。

# COMMAND ----------

# MAGIC %md
# MAGIC ## ④ 记录模型（models from code）
# MAGIC `resources` 声明 Agent 运行时要访问的 Databricks 资源，部署时系统会**自动配置凭证**（automatic authentication passthrough）。
# MAGIC 漏声明 → 部署后调用工具报 403。

# COMMAND ----------

from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint, DatabricksTable

# legacy 模式的 LLM 是 serving 端点，需要声明；gateway 模式的 system.ai.* 模型不是 serving 端点，不能用 DatabricksServingEndpoint 声明
llm_resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT)] if LLM_MODE == "legacy" else []

resources = [
    *llm_resources,
    DatabricksServingEndpoint(endpoint_name=RISK_ENDPOINT),
    *[DatabricksFunction(function_name=fn) for fn in [FN_PROFILE, FN_FEATURES, FN_ACTIONS, FN_POLICY]],
    *[DatabricksTable(table_name=t) for t in [VEHICLES_TABLE, FEATURE_TABLE, ACTIONS_TABLE, POLICY_TABLE]],
]

input_example = {"input": [{"role": "user", "content": "车辆 V00042 的电池需要检修吗？"}]}

with mlflow.start_run(run_name="battery_service_agent"):
    logged = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        model_config="agent_config.yaml",
        input_example=input_example,
        resources=resources,
        pip_requirements=[
            "mlflow[databricks]>=3.1",
            "databricks-sdk",
            "openai",
            "pyyaml",
            "unitycatalog-ai[databricks]",
        ],
    )
print("model_uri:", logged.model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⑤ 部署前校验
# MAGIC 在一个全新的虚拟环境里加载模型并预测，提前发现“依赖缺失”之类的问题（比部署失败再查 Build logs 快得多）。

# COMMAND ----------

mlflow.models.predict(model_uri=logged.model_uri, input_data=input_example, env_manager="uv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⑥ 注册到 Unity Catalog

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mv = mlflow.register_model(logged.model_uri, AGENT_MODEL_NAME)
print(f"✅ {AGENT_MODEL_NAME} Version {mv.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⑦ 部署：`agents.deploy`
# MAGIC 一行代码完成：
# MAGIC - 创建 Serving 端点（scale-to-zero）；
# MAGIC - 开启推理表 + MLflow Trace 记录（生产监控）；
# MAGIC - 创建 **Review App**（给售后专家试用、打分、反馈的网页）。
# MAGIC
# MAGIC 需要约 15 分钟。
# MAGIC
# MAGIC ⚠️ **关于 Unity Gateway（2026-09 的平台变化）**：本地测试（② ③）两种模式都能用。
# MAGIC 部署到 Model Serving 后，Agent 用自动鉴权透传的短期凭证访问资源；该凭证能否调用 Unity Gateway 的 `system.ai.*` 模型取决于你的 workspace 设置。
# MAGIC 如果部署后调用报 401/403：
# MAGIC 1. 最简单：`00_config` 改 `LLM_MODE = "legacy"`，重新运行 ① ④ ⑥ ⑦；
# MAGIC 2. 或按官方推荐把 Agent 部署到 **Databricks Apps**（文档 7.7 有说明）。

# COMMAND ----------

from databricks import agents

deployment = agents.deploy(AGENT_MODEL_NAME, mv.version, tags={"project": "ev-battery-lab"}, scale_to_zero=True)
print("endpoint  :", deployment.endpoint_name)
print("review app:", deployment.review_app_url)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⑧ 调用已部署的 Agent（端点 Ready 之后运行）

# COMMAND ----------

from databricks.sdk import WorkspaceClient

client = WorkspaceClient().serving_endpoints.get_open_ai_client()
r = client.responses.create(
    model=deployment.endpoint_name,
    input=[{"role": "user", "content": f"车辆 {DEGRADED_VID} 需要进站吗？在质保内吗？"}],
)
print(r.output_text if hasattr(r, "output_text") else r)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✍️ 练习
# MAGIC 1. 把 `LLM_ENDPOINT` 换成另一个模型，重跑 ③，比较两个模型的评估得分 —— 用数据选模型。
# MAGIC 2. 新增工具 `get_fleet_summary(owner_name)`（UC 函数：某车队所有车辆的风险概况），并加入评估集。
# MAGIC 3. 把 Review App 链接发给同事，让他们提问并打分；然后在 Experiments 中查看收集到的反馈。
