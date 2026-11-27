# Databricks notebook source
# MAGIC %md
# MAGIC # 99 · 清理资源
# MAGIC
# MAGIC ⚠️ 删除操作不可恢复。默认只删除**会持续计费**的资源：阶段 9 的 Serving 端点和在线特征存储。
# MAGIC
# MAGIC | 开关 | 默认 | 作用 |
# MAGIC |---|---|---|
# MAGIC | `DELETE_RAW_FILES` | False | 删除 Volume 里约 7.7 GB 的原始 .mat 文件（解析结果已在 Delta 表中） |
# MAGIC | `DROP_SCHEMA` | False | 删除整个 `battery_life_lab`：所有表、视图、模型、Volume |
# MAGIC
# MAGIC 需要在界面里手动处理的：阶段 6 创建的 **Job**（暂停或删除）、阶段 8 创建的 **Genie Agent**。

# COMMAND ----------

# MAGIC %pip install -U -q databricks-sdk databricks-feature-engineering

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

DELETE_RAW_FILES = False
DROP_SCHEMA = False

from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

w = WorkspaceClient()
fe = FeatureEngineeringClient()

# 阶段 9：Serving 端点
try:
    w.serving_endpoints.delete(SERVING_ENDPOINT)
    print("🗑️ 已删除 Serving 端点", SERVING_ENDPOINT)
except NotFound:
    print("没有 Serving 端点，跳过")

# 阶段 9：在线特征存储（常驻计算资源，会持续计费）
if fe.get_online_store(name=ONLINE_STORE_NAME) is not None:
    fe.delete_online_store(name=ONLINE_STORE_NAME)
    print("🗑️ 已删除在线特征存储", ONLINE_STORE_NAME)
else:
    print("没有在线特征存储，跳过")

# COMMAND ----------

import os

if DELETE_RAW_FILES and not DROP_SCHEMA and os.path.isdir(RAW_VOLUME_PATH):
    for fname in os.listdir(RAW_VOLUME_PATH):
        os.remove(f"{RAW_VOLUME_PATH}/{fname}")
        print("🗑️ 已删除", fname)

if DROP_SCHEMA:
    spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE")
    print(f"🗑️ 已删除 schema {CATALOG}.{SCHEMA}")
else:
    print("保留 schema。如需删除，设置 DROP_SCHEMA = True 后重跑本单元。")
