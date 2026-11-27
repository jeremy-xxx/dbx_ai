# Databricks notebook source
# MAGIC %md
# MAGIC # 99 · 清理资源
# MAGIC
# MAGIC ⚠️ 删除操作不可恢复。默认只删除 **会持续计费** 的 Serving 端点和 AI Search 端点（阶段 7c 可选创建）；
# MAGIC 如需删除整个 schema（表、模型、函数、Volume），把 `DROP_SCHEMA` 改为 True。
# MAGIC
# MAGIC Agent Bricks 创建的 Knowledge Assistant、定时 Job 请在 UI 中手动删除。

# COMMAND ----------

# MAGIC %pip install -U -q databricks-sdk databricks-ai-search

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

# 阶段 7c（可选）：AI Search 索引和端点。端点是常驻计算资源，会持续计费
from databricks.ai_search.client import AISearchClient

search_client = AISearchClient(disable_notice=True)
if search_client.endpoint_exists(AI_SEARCH_ENDPOINT):
    if search_client.index_exists(endpoint_name=AI_SEARCH_ENDPOINT, index_name=AI_SEARCH_INDEX):
        search_client.delete_index(endpoint_name=AI_SEARCH_ENDPOINT, index_name=AI_SEARCH_INDEX)
        print("🗑️ 已删除 AI Search 索引", AI_SEARCH_INDEX)
    search_client.delete_endpoint(AI_SEARCH_ENDPOINT)
    print("🗑️ 已删除 AI Search 端点", AI_SEARCH_ENDPOINT)
else:
    print("没有创建过 AI Search 端点，跳过")

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
