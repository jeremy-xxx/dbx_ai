# Databricks notebook source
# MAGIC %md
# MAGIC # 99 · 清理资源
# MAGIC
# MAGIC ⚠️ 删除操作不可恢复。默认只删除 **会持续计费** 的 Serving 端点；
# MAGIC 如需删除整个 schema（表、模型、函数、Volume），把 `DROP_SCHEMA` 改为 True。
# MAGIC
# MAGIC Agent Bricks 创建的 Knowledge Assistant、定时 Job 请在 UI 中手动删除。

# COMMAND ----------

# MAGIC %pip install -U -q databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

DROP_SCHEMA = False

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
targets = [
    e.name for e in w.serving_endpoints.list()
    if e.name == RISK_ENDPOINT
    or e.name == RISK_ENDPOINT.replace("battery-risk", "ext-gpt")
    or (e.name.startswith("agents_") and f"{SCHEMA}-battery_service_agent" in e.name)
]
print("将删除以下端点:", targets)

# COMMAND ----------

for name in targets:
    w.serving_endpoints.delete(name)
    print("🗑️ 已删除端点", name)

# COMMAND ----------

# 阶段 9 的原始 .mat 文件约 8 GB；解析结果已在 Delta 表中，原始文件可单独删除
DELETE_RAW_BATTERY_FILES = False
if DELETE_RAW_BATTERY_FILES and not DROP_SCHEMA:
    import os
    if os.path.isdir(RAW_VOLUME_PATH):
        for fname in os.listdir(RAW_VOLUME_PATH):
            os.remove(f"{RAW_VOLUME_PATH}/{fname}")
            print("🗑️ 已删除", fname)

if DROP_SCHEMA:
    spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE")
    print(f"🗑️ 已删除 schema {CATALOG}.{SCHEMA}")
else:
    print("保留 schema。如需删除，设置 DROP_SCHEMA = True 后重跑本单元。")
