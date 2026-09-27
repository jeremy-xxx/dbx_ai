# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Models：注册到 Unity Catalog、别名与车队批量打分
# MAGIC
# MAGIC 对应文档：`docs/03-models.md`

# COMMAND ----------

# MAGIC %pip install -U -q "mlflow[databricks]>=3.1" scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import recall_score, roc_auc_score

# （界面无变化）决定下面 register_model 注册到 Unity Catalog
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
# （界面无变化）ev-battery-risk 已在阶段 2 创建，这里只是为了按实验查询 Run
exp = mlflow.set_experiment(EXPERIMENT_PATH)

runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="tags.candidate = 'true'",
    order_by=["metrics.roc_auc DESC"],
)
assert len(runs) >= 2, "请先运行 02_experiments_tracking"
best, second = runs.iloc[0], runs.iloc[1]
print("第一名:", best["tags.mlflow.runName"], round(best["metrics.roc_auc"], 4))
print("第二名:", second["tags.mlflow.runName"], round(second["metrics.roc_auc"], 4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 先把“第二名” Run 的模型注册为 Version 1，并设为 champion
# MAGIC （模拟“线上已有一个旧模型”的场景）

# COMMAND ----------

# 👀 界面：左侧 Models（或 Catalog → main → ev_battery_lab → Models）出现 battery_risk_model，含 Version 1
mv1 = mlflow.register_model(f"runs:/{second.run_id}/model", BASE_MODEL_NAME)
# 👀 界面：模型页面 Version 1 旁显示别名标签 @champion
client.set_registered_model_alias(BASE_MODEL_NAME, "champion", mv1.version)

# 👀 界面：模型页面 Overview 的描述区显示这段 Markdown
client.update_registered_model(
    BASE_MODEL_NAME,
    description=(
        "## 动力电池 90 天故障风险模型\n"
        "- 输入：车辆档案 + BMS 遥测特征（见 signature）\n"
        "- 输出：`[P(正常), P(故障)]`\n"
        "- 风险等级：p<0.3 low / 0.3–0.6 medium / ≥0.6 high（见《电池维保规程》）\n"
        "- Owner：电池售后数据组\n"
    ),
)
client.update_model_version(BASE_MODEL_NAME, mv1.version, description=f"来自 run {second['tags.mlflow.runName']}")
print(f"✅ 注册 Version {mv1.version} 并设为 champion")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 把“第一名” Run 的模型注册为 Version 2，并设为 challenger

# COMMAND ----------

# 👀 界面：同一模型下新增 Version 2，下一行给它加上 @challenger 标签
mv2 = mlflow.register_model(f"runs:/{best.run_id}/model", BASE_MODEL_NAME)
client.set_registered_model_alias(BASE_MODEL_NAME, "challenger", mv2.version)
client.update_model_version(BASE_MODEL_NAME, mv2.version, description=f"来自 run {best['tags.mlflow.runName']}")
print(f"✅ 注册 Version {mv2.version} 并设为 challenger")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Champion vs Challenger：AUC 更高 且 recall 不下降，才上位
# MAGIC 真实项目中评估集应是独立的最新数据；这里用车龄最大的 1000 辆车模拟“老化车队”。

# COMMAND ----------

eval_df = spark.sql(f"SELECT * FROM {VEHICLES_TABLE} ORDER BY vehicle_age_months DESC LIMIT 1000").toPandas()
X_eval = eval_df[MODEL_FEATURES].astype({c: "float64" for c in NUMERIC_FEATURES})
y_eval = eval_df[LABEL]
THRESHOLD = 0.3   # 售后使用的“中风险”起点

def score_alias(alias: str) -> dict:
    m = mlflow.pyfunc.load_model(f"models:/{BASE_MODEL_NAME}@{alias}")
    proba = pd.DataFrame(m.predict(X_eval)).iloc[:, 1]
    return {"auc": roc_auc_score(y_eval, proba), "recall@0.3": recall_score(y_eval, proba >= THRESHOLD)}

champ, chall = score_alias("champion"), score_alias("challenger")
print("champion  :", {k: round(v, 4) for k, v in champ.items()})
print("challenger:", {k: round(v, 4) for k, v in chall.items()})

if chall["auc"] > champ["auc"] and chall["recall@0.3"] >= champ["recall@0.3"]:
    # 👀 界面：@champion 标签从 Version 1 移到 Version 2，@challenger 被删除
    client.set_registered_model_alias(BASE_MODEL_NAME, "champion", mv2.version)
    client.delete_registered_model_alias(BASE_MODEL_NAME, "challenger")
    client.set_model_version_tag(BASE_MODEL_NAME, mv2.version, "validation_status", "approved")
    print(f"🏆 challenger 通过闸门，champion → Version {mv2.version}")
else:
    client.set_model_version_tag(BASE_MODEL_NAME, mv2.version, "validation_status", "rejected")
    print("⛔ challenger 未通过闸门（AUC 未提升或漏报变多），champion 保持不变")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 用别名加载模型

# COMMAND ----------

champion = client.get_model_version_by_alias(BASE_MODEL_NAME, "champion")
print(f"champion = Version {champion.version}, run_id = {champion.run_id}")

model = mlflow.pyfunc.load_model(f"models:/{BASE_MODEL_NAME}@champion")
print("输入签名:", model.metadata.get_input_schema())
sample = X_eval.head(5)
display(sample.assign(fault_proba=pd.DataFrame(model.predict(sample)).iloc[:, 1].round(3).values))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Spark 分布式批量打分：整个车队的风险分布

# COMMAND ----------

from pyspark.sql import functions as F

udf = mlflow.pyfunc.spark_udf(spark, f"models:/{BASE_MODEL_NAME}@champion", result_type="array<double>")

scored = (spark.table(VEHICLES_TABLE)
    .withColumn("proba", udf(*[F.col(c).cast("double") if c in NUMERIC_FEATURES else F.col(c) for c in MODEL_FEATURES]))
    .withColumn("fault_proba", F.round(F.col("proba")[1], 4))
    .withColumn("risk_level", F.when(F.col("fault_proba") >= 0.6, "high")
                               .when(F.col("fault_proba") >= 0.3, "medium").otherwise("low"))
    .select("vehicle_id", "vehicle_model", "battery_chemistry", "soh_pct", LABEL, "fault_proba", "risk_level"))

display(scored.groupBy("vehicle_model").pivot("risk_level", ["low", "medium", "high"]).count())
display(scored.orderBy(F.desc("fault_proba")).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✍️ 练习
# MAGIC 1. 把 `scored` 写成 `fleet_risk_scores` 表，去 Catalog 看它的血缘。
# MAGIC 2. 回滚演练：`client.set_registered_model_alias(BASE_MODEL_NAME, "champion", 1)`，再加载验证。
# MAGIC 3. 在 UI 的 **Models → battery_risk_model → Permissions** 查看可授予的权限。
