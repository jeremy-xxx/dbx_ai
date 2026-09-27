# Databricks notebook source
# MAGIC %md
# MAGIC # 08 · 综合项目：车队电池风险巡检流水线
# MAGIC
# MAGIC 对应文档：`docs/08-capstone.md`
# MAGIC
# MAGIC ```
# MAGIC 特征表 ──► 风险模型批量打分 ──► 风险分层 + 安全红线 ──► 匹配处置动作 + 质保判断
# MAGIC                                                         │
# MAGIC                         ai_query 批量生成车主/车队进站通知 ◄┤
# MAGIC                         Top N 交给 Agent 生成技师工单建议 ◄─┘
# MAGIC                                         ↓
# MAGIC                         battery_service_campaign 表（给工单系统 / 仪表盘使用）
# MAGIC ```
# MAGIC 本 notebook 设计成可以直接放进 **Jobs（工作流）** 每天定时运行。

# COMMAND ----------

# MAGIC %pip install -U -q "mlflow[databricks]>=3.1" databricks-sdk openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

dbutils.widgets.text("top_n", "20", "生成通知的车辆数")
dbutils.widgets.text("agent_top_n", "3", "交给 Agent 生成工单的车辆数")
dbutils.widgets.text("agent_endpoint", "", "Agent 端点名（留空自动查找）")
TOP_N = int(dbutils.widgets.get("top_n"))
AGENT_TOP_N = int(dbutils.widgets.get("agent_top_n"))
CAMPAIGN_TABLE = f"{CATALOG}.{SCHEMA}.battery_service_campaign"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 批量打分（Models + Features）+ 安全红线
# MAGIC 从特征表读取最新遥测，用 `battery_risk_model@champion` 打分。红线判断是确定性规则，不依赖模型。

# COMMAND ----------

import mlflow
from pyspark.sql import functions as F

# （界面无变化）用于按别名加载 UC 中的模型
mlflow.set_registry_uri("databricks-uc")
udf = mlflow.pyfunc.spark_udf(spark, f"models:/{BASE_MODEL_NAME}@champion", result_type="array<double>")

scored = (spark.table(FEATURE_TABLE)
    .withColumn("fault_proba", F.round(udf(*MODEL_FEATURES)[1], 4))
    .withColumn("safety_redline",
                (F.col("max_cell_temp_c_30d") >= REDLINE_TEMP_C) | (F.col("cell_voltage_imbalance_mv") >= REDLINE_IMBALANCE_MV))
    .withColumn("risk_level",
        F.when(F.col("fault_proba") >= 0.6, "high")
         .when(F.col("fault_proba") >= 0.3, "medium")
         .otherwise("low")))

display(scored.groupBy("vehicle_model", "risk_level").agg(
    F.count("*").alias("n"), F.sum(F.col("safety_redline").cast("int")).alias("redline")
).orderBy("vehicle_model", "risk_level"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 选出需要处置的车辆，关联档案、质保、处置动作
# MAGIC 排序规则：**红线车永远排最前**（P0），然后按故障概率。

# COMMAND ----------

vehicles = spark.table(VEHICLES_TABLE).select(
    "vehicle_id", "owner_name", "owner_type", "odometer_km",
    F.expr("""CASE WHEN vehicle_model = 'commercial_van'
                   THEN vehicle_age_months < 60 AND odometer_km < 300000
                   ELSE vehicle_age_months < 96 AND odometer_km < 150000 END""").alias("in_warranty"),
)
actions = spark.table(ACTIONS_TABLE).withColumnRenamed("vehicle_model", "a_model").withColumnRenamed("risk_level", "a_risk")

candidates = (scored.where("risk_level = 'high' OR safety_redline")
    .join(vehicles, "vehicle_id")
    # 红线车按 high 取动作，但优先级提升为 P0
    .withColumn("action_risk", F.when(F.col("safety_redline"), "high").otherwise(F.col("risk_level")))
    .join(actions, (F.col("vehicle_model") == F.col("a_model")) & (F.col("action_risk") == F.col("a_risk")))
    .withColumn("priority", F.when(F.col("safety_redline"), "P0").otherwise(F.col("priority")))
    .select("vehicle_id", "owner_name", "owner_type", "vehicle_model", "battery_chemistry", "climate_zone",
            "vehicle_age_months", "odometer_km", "soh_pct", "max_cell_temp_c_30d", "cell_voltage_imbalance_mv",
            "bms_alerts_90d", "fast_charge_ratio", "fault_proba", "risk_level", "safety_redline",
            "in_warranty", "priority", "action", "service_hours")
    .orderBy(F.desc("safety_redline"), F.desc("fault_proba"))
    .limit(TOP_N))

candidates.createOrReplaceTempView("candidates")
display(candidates)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. `ai_query` 批量生成进站通知（Serving + Foundation Model）
# MAGIC 把规程要点写进 prompt：红线车必须清楚说明安全注意事项；不透露概率；不用恐慌措辞；车队客户对象是调度负责人。

# COMMAND ----------

# 👀 界面：这些调用计入 AI Gateway → Models → 你的模型 的 Total tokens / Metrics
notices = spark.sql(f"""
SELECT *,
  ai_query(
    '{LLM_MODEL}',
    concat(
      '你是电动车公司的售后服务顾问，请给下面的车辆写一条中文进站通知（不超过150字）。\\n',
      '规则：\\n',
      '1) 若 owner_type=fleet，对象是车队调度负责人，需写明车辆ID和建议停运安排；否则称呼车主姓名；\\n',
      '2) 若 safety_redline=true：必须明确提醒立即停止使用快充、避免长时间高速行驶、24小时内进站，并提供救援拖车服务；\\n',
      '3) 否则按以下处置动作和时限邀请进站：', action, '；\\n',
      '4) 若 in_warranty=true，说明本次检测免费；\\n',
      '5) 不得提及任何故障概率或内部评分，不得使用“起火”“报废”“爆炸”等恐慌性措辞；语气专业、平和。\\n',
      '车辆信息：', to_json(named_struct(
        'vehicle_id', vehicle_id, 'owner_name', owner_name, 'owner_type', owner_type,
        'vehicle_model', vehicle_model, 'safety_redline', safety_redline,
        'in_warranty', in_warranty, 'priority', priority))
    )
  ) AS notice_draft
FROM candidates
""")
display(notices.select("vehicle_id", "owner_name", "owner_type", "priority", "safety_redline", "in_warranty", "notice_draft"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Top N 车辆交给 Agent 生成技师工单建议（Agents）
# MAGIC 调用阶段 7 部署的 Agent 端点。若尚未部署，这一步会跳过。

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
# agents.deploy 生成的端点名形如 agents_<catalog>-<schema>-battery_service_agent
agent_endpoint = dbutils.widgets.get("agent_endpoint") or next(
    (e.name for e in w.serving_endpoints.list()
     if e.name.startswith("agents_") and f"{SCHEMA}-battery_service_agent" in e.name),
    None,
)
print("Agent 端点:", agent_endpoint)

advice_rows = []
if agent_endpoint:
    oai = w.serving_endpoints.get_open_ai_client()
    for row in candidates.limit(AGENT_TOP_N).collect():
        try:
            # 👀 界面：Agent 端点页面 Metrics 请求数增加，推理表中留下请求记录
            r = oai.responses.create(
                model=agent_endpoint,
                input=[{"role": "user", "content":
                        f"请为车辆 {row.vehicle_id} 生成技师工单建议：检测项目、处置步骤、是否走质保、需要谁审批"}],
            )
            advice_rows.append((row.vehicle_id, r.output_text))
        except Exception as e:
            advice_rows.append((row.vehicle_id, f"Agent 调用失败: {e}"))

advice_df = spark.createDataFrame(advice_rows or [("", "")], "vehicle_id STRING, agent_work_order STRING")
display(advice_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 写入结果表

# COMMAND ----------

champion_version = mlflow.MlflowClient().get_model_version_by_alias(BASE_MODEL_NAME, "champion").version
campaign = (notices.join(advice_df, "vehicle_id", "left")
    .withColumn("run_date", F.current_date())
    .withColumn("model_version", F.lit(str(champion_version))))

# 👀 界面：Tables 出现 battery_service_campaign；每次运行追加（append）一批，按 run_date 区分
campaign.write.mode("append").option("mergeSchema", "true").saveAsTable(CAMPAIGN_TABLE)
print(f"✅ 已写入 {CAMPAIGN_TABLE}")
display(spark.table(CAMPAIGN_TABLE).orderBy(F.desc("run_date"), "priority", F.desc("fault_proba")).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 运营看板查询（可以粘到 SQL Editor 做 Dashboard）

# COMMAND ----------

display(spark.sql(f"""
SELECT run_date, priority, vehicle_model,
       count(*) AS vehicles,
       sum(CASE WHEN in_warranty THEN 1 ELSE 0 END) AS in_warranty,
       round(sum(service_hours), 1) AS total_service_hours
FROM {CAMPAIGN_TABLE}
GROUP BY ALL ORDER BY run_date DESC, priority, vehicle_model
"""))
