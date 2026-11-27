# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · 效果监控：真实寿命出来后，模型到底准不准？
# MAGIC
# MAGIC 对应文档：`docs/07-monitoring.md`
# MAGIC
# MAGIC **业务场景**：阶段 6 给新批次（batch3）打分后，这批电芯继续测试，几个月后真实寿命陆续出来。
# MAGIC 本 notebook 模拟“真实寿命回填”，检查三件事：
# MAGIC
# MAGIC | 检查 | 问题 |
# MAGIC |---|---|
# MAGIC | ① 预测误差 | 新批次的误差和上线时测试集的误差比，变大了多少？ |
# MAGIC | ② 质检决策质量 | 真正的短寿命电芯，有没有被标记出来？误判了多少正常电芯？ |
# MAGIC | ③ 特征漂移 | 新批次的特征分布和训练数据一样吗？误差变大的原因在哪？ |
# MAGIC
# MAGIC 结果写入 `model_monitoring` 表，每次运行一行，可以长期追踪。

# COMMAND ----------

# MAGIC %pip install -U -q "mlflow[databricks]>=3.1"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
champion = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
test_mape = float(champion.tags["test_mape"])
model_features = champion.tags["features"].split(",")
print(f"监控 {MODEL_NAME} v{champion.version}：上线时测试集误差 {test_mape:.1f}%，使用 {len(model_features)} 个特征")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 回填真实寿命，计算误差
# MAGIC 每个电芯取当前模型版本的最新一次预测，和 `cell_metadata` 里的真实寿命对比。
# MAGIC （真实场景里，真实寿命来自之后的测试记录；本项目数据集里已经有了，所以直接关联。）

# COMMAND ----------

evaluated = spark.sql(f"""
    SELECT p.cell_id, m.dataset_role, m.charge_policy, m.cycle_life AS actual_cycle_life,
           p.predicted_cycle_life, p.predicted_low, p.predicted_high, p.qc_decision
    FROM {PREDICTION_TABLE} p JOIN {CELL_META_TABLE} m USING (cell_id)
    WHERE p.model_version = '{champion.version}'
    QUALIFY row_number() OVER (PARTITION BY p.cell_id ORDER BY p.scored_at DESC) = 1
""").toPandas()
assert len(evaluated) > 0, "没有预测记录，请先运行 06_batch_score"

evaluated["abs_pct_error"] = (evaluated.predicted_cycle_life - evaluated.actual_cycle_life).abs() / evaluated.actual_cycle_life * 100
evaluated["in_range"] = evaluated.actual_cycle_life.between(evaluated.predicted_low, evaluated.predicted_high)
evaluated["actually_short"] = evaluated.actual_cycle_life < SHORT_LIFE_THRESHOLD

by_role = evaluated.groupby("dataset_role").agg(
    cells=("cell_id", "count"),
    mape=("abs_pct_error", "mean"),
    in_range_rate=("in_range", "mean"),
).round(3)
display(by_role.reset_index())

new = evaluated[evaluated.dataset_role == "new_batch"]
new_mape = float(new.abs_pct_error.mean()) if len(new) else float("nan")
ratio = new_mape / test_mape
print(f"新批次误差 {new_mape:.1f}%，是上线时测试集误差（{test_mape:.1f}%）的 {ratio:.2f} 倍")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 质检决策质量
# MAGIC 对质检来说，最关心的是**漏检**：真正的短寿命电芯被当成正常放行了。

# COMMAND ----------

flagged = new.qc_decision.isin(["short_life", "borderline"])   # 被要求复检或延长测试
caught = int((flagged & new.actually_short).sum())
missed = int((~flagged & new.actually_short).sum())
false_alarm = int((flagged & ~new.actually_short).sum())
print(f"真实短寿命电芯 {int(new.actually_short.sum())} 个：抓到 {caught} 个，漏检 {missed} 个")
print(f"被标记复检的电芯中，实际寿命正常的有 {false_alarm} 个（多做了检测，但不会出质量问题）")
display(pd.crosstab(new.qc_decision, new.actually_short.map({True: "实际短寿命", False: "实际正常"})))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 特征漂移：新批次和训练数据像不像？
# MAGIC 对模型用到的每个特征，比较新批次和训练集的分布：
# MAGIC - **标准化均值差**：（新批次均值 − 训练均值）÷ 训练标准差，绝对值越大差得越多，一般超过 0.5 就值得注意；
# MAGIC - **PSI**（群体稳定性指数）：按训练集分位数分箱后比较两边占比，常用经验阈值是 < 0.1 稳定、0.1–0.25 轻微漂移、> 0.25 明显漂移。

# COMMAND ----------

features = spark.table(FEATURE_TABLE).toPandas().merge(
    spark.table(CELL_META_TABLE).select("cell_id", "dataset_role").toPandas(), on="cell_id")
train_f = features[features.dataset_role == "train"]
new_f = features[features.dataset_role == "new_batch"]

def psi(expected: pd.Series, actual: pd.Series, bins: int = 5) -> float:
    edges = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected.dropna(), edges)[0] / expected.notna().sum()
    a = np.histogram(actual.dropna(), edges)[0] / actual.notna().sum()
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))

drift = pd.DataFrame([{
    "feature": col,
    "train_mean": train_f[col].mean(),
    "new_batch_mean": new_f[col].mean(),
    "std_mean_diff": (new_f[col].mean() - train_f[col].mean()) / train_f[col].std(),
    "psi": psi(train_f[col], new_f[col]),
} for col in model_features]).sort_values("psi", ascending=False)
display(drift.round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 结论与告警，写入监控表

# COMMAND ----------

alert = ratio > DRIFT_ALERT_RATIO
top = drift.iloc[0]
monitoring_row = pd.DataFrame([{
    "monitored_at": pd.Timestamp.now(tz="UTC"),
    "model_version": str(champion.version),
    "new_batch_cells": len(new),
    "test_mape": test_mape,
    "new_batch_mape": new_mape,
    "error_ratio": ratio,
    "in_range_rate": float(new.in_range.mean()),
    "short_life_missed": missed,
    "top_drift_feature": top.feature,
    "top_drift_psi": float(top.psi),
    "alert": bool(alert),
}])
# 👀 界面：Catalog → main → battery_life_lab → Tables 出现 model_monitoring；每次运行追加一行
spark.createDataFrame(monitoring_row).write.mode("append").option("mergeSchema", "true").saveAsTable(MONITORING_TABLE)

if alert:
    print(f"🚨 告警：新批次误差是测试集的 {ratio:.2f} 倍（阈值 {DRIFT_ALERT_RATIO}）。"
          f"漂移最大的特征是 {top.feature}（PSI {top.psi:.2f}）。建议用新批次的部分数据重新训练（见练习 1）。")
else:
    print(f"✅ 新批次误差在可接受范围内（{ratio:.2f} 倍 ≤ {DRIFT_ALERT_RATIO}）")
display(spark.table(MONITORING_TABLE).orderBy("monitored_at"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✍️ 练习
# MAGIC 1. **再训练闭环**：在 `01_ingest` 第 ④ 节把 batch3 的一半电芯改为 `train`（例如 `b3c` 编号为偶数的），
# MAGIC    依次重跑 01 → 03 → 04 → 05 → 06 → 07。新模型能过闸门吗？新批次剩下的电芯误差有没有降低？
# MAGIC 2. 找出新批次里误差最大的 3 个电芯，看它们的快充策略。训练集里有类似的策略吗？
# MAGIC 3. 在左侧 **SQL Editor** 里查询 `model_monitoring`，做一个简单的 Dashboard，追踪 `error_ratio` 随时间的变化。
