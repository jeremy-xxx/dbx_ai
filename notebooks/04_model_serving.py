# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Serving：部署电池风险模型为 REST 端点
# MAGIC
# MAGIC 对应文档：`docs/04-serving.md`
# MAGIC
# MAGIC ⚠️ 端点运行会计费，已开启 scale-to-zero；学完请运行 `99_cleanup`。

# COMMAND ----------

# MAGIC %pip install -U -q "mlflow[databricks]>=3.1" databricks-sdk matplotlib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from datetime import timedelta

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
w = WorkspaceClient()
champion_version = MlflowClient().get_model_version_by_alias(BASE_MODEL_NAME, "champion").version
print(f"将部署 {BASE_MODEL_NAME} Version {champion_version} → 端点 {RISK_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 创建（或更新）端点
# MAGIC 需要 10–20 分钟。等待期间去 **Serving → 你的端点 → Logs → Build logs** 看构建过程。

# COMMAND ----------

served = [ServedEntityInput(
    name="battery-risk-champion",
    entity_name=BASE_MODEL_NAME,
    entity_version=str(champion_version),
    workload_size="Small",
    scale_to_zero_enabled=True,
)]

try:
    w.serving_endpoints.get(RISK_ENDPOINT)
    print("端点已存在 → 更新配置")
    w.serving_endpoints.update_config_and_wait(
        name=RISK_ENDPOINT, served_entities=served, timeout=timedelta(minutes=40))
except NotFound:
    print("创建新端点 …")
    w.serving_endpoints.create_and_wait(
        name=RISK_ENDPOINT,
        config=EndpointCoreConfigInput(served_entities=served),
        timeout=timedelta(minutes=40),
    )

ep = w.serving_endpoints.get(RISK_ENDPOINT)
print("状态:", ep.state.ready, "| 配置更新:", ep.state.config_update)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 调用方式 ①：SDK

# COMMAND ----------

records = [
    # 高风险：高温地区、高快充占比、NMC、SOH 低、压差大的物流车
    {"vehicle_age_months": 60.0, "odometer_km": 380000.0, "cycle_count": 1200.0, "fast_charge_ratio": 0.75,
     "avg_depth_of_discharge": 0.85, "max_cell_temp_c_30d": 58.0, "cell_voltage_imbalance_mv": 130.0,
     "bms_alerts_90d": 5.0, "soh_pct": 72.0,
     "vehicle_model": "commercial_van", "battery_chemistry": "NMC", "climate_zone": "hot"},
    # 低风险：温和地区、慢充为主、LFP 的新城市小车
    {"vehicle_age_months": 12.0, "odometer_km": 15000.0, "cycle_count": 70.0, "fast_charge_ratio": 0.1,
     "avg_depth_of_discharge": 0.5, "max_cell_temp_c_30d": 36.0, "cell_voltage_imbalance_mv": 12.0,
     "bms_alerts_90d": 0.0, "soh_pct": 98.5,
     "vehicle_model": "city_ev", "battery_chemistry": "LFP", "climate_zone": "temperate"},
]

resp = w.serving_endpoints.query(name=RISK_ENDPOINT, dataframe_records=records)
for r, p in zip(records, resp.predictions):
    print(f"{r['vehicle_model']:<15} SOH={r['soh_pct']:>5}  → 90 天故障概率 {p[1]:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 调用方式 ②：原生 REST（任何语言都能照抄）

# COMMAND ----------

import requests

url = f"{w.config.host}/serving-endpoints/{RISK_ENDPOINT}/invocations"
headers = {**w.config.authenticate(), "Content-Type": "application/json"}
r = requests.post(url, headers=headers, json={"dataframe_records": records}, timeout=120)
print(r.status_code, r.json())

# dataframe_split 格式
split_payload = {"dataframe_split": {"columns": list(records[0].keys()), "data": [list(x.values()) for x in records]}}
print(requests.post(url, headers=headers, json=split_payload, timeout=120).json())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 调用方式 ③：SQL `ai_query`
# MAGIC 对自定义模型端点，需要用 `returnType` 声明返回类型。

# COMMAND ----------

num_struct = ", ".join(f"'{c}', CAST({c} AS DOUBLE)" for c in NUMERIC_FEATURES)
cat_struct = ", ".join(f"'{c}', {c}" for c in CATEGORICAL_FEATURES)

display(spark.sql(f"""
SELECT vehicle_id, vehicle_model, soh_pct, {LABEL},
       ai_query('{RISK_ENDPOINT}', named_struct({num_struct}, {cat_struct}),
                returnType => 'ARRAY<DOUBLE>')[1] AS fault_proba
FROM {VEHICLES_TABLE}
ORDER BY soh_pct
LIMIT 10
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. What-if 分析：快充占比对风险的影响
# MAGIC 固定其它特征，只改变 `fast_charge_ratio`，一次请求批量发送。

# COMMAND ----------

import matplotlib.pyplot as plt

base = {**records[0], "max_cell_temp_c_30d": 45.0, "cell_voltage_imbalance_mv": 60.0, "soh_pct": 85.0}
ratios = [round(x * 0.1, 1) for x in range(0, 11)]
batch = [{**base, "fast_charge_ratio": r} for r in ratios]
probs = [p[1] for p in w.serving_endpoints.query(name=RISK_ENDPOINT, dataframe_records=batch).predictions]

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(ratios, probs, marker="o")
ax.set_xlabel("fast_charge_ratio"); ax.set_ylabel("P(fault in 90d)"); ax.set_title("What-if: fast charging")
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 故意发送错误请求，观察 schema 校验

# COMMAND ----------

bad = [{**records[0], "soh_pct": "abc"}]
r = requests.post(url, headers=headers, json={"dataframe_records": bad}, timeout=120)
print(r.status_code, r.text[:500])

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✍️ 练习
# MAGIC 1. 仿照第 5 节，对 `max_cell_temp_c_30d`（30→65℃）做 What-if 分析。
# MAGIC 2. 在 UI 里做 90/10 流量切分（见文档 4.6）。
# MAGIC 3. 用本地终端 + Personal Access Token 跑一次 curl（见文档 4.4）。
