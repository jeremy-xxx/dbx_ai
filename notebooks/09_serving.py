# Databricks notebook source
# MAGIC %md
# MAGIC # 09 ·（可选）部署成实时 API
# MAGIC
# MAGIC 对应文档：`docs/09-serving.md`
# MAGIC
# MAGIC **什么时候需要实时 API？** 主线的批量预测（阶段 6）已经满足“一批电芯测完后统一打分”。
# MAGIC 如果测试台软件希望**每个电芯测完 100 个循环就立刻拿到预测**，就需要一个随时可调用的接口。
# MAGIC
# MAGIC 模型是用 `fe.log_model` 保存的，调用时只传电芯 ID，所以端点需要**实时查特征**：
# MAGIC
# MAGIC ```
# MAGIC 测试台软件 ──{"cell_id": "b3c5"}──► Serving 端点 ──查特征──► 在线特征存储（Lakebase）
# MAGIC                                         │                         ▲
# MAGIC                                         ▼                         │ publish_table 同步
# MAGIC                                  返回 log10(寿命)            特征表 cell_early_features
# MAGIC ```
# MAGIC
# MAGIC ⚠️ **会创建两个持续计费的资源**：在线特征存储、Serving 端点（已开启空闲缩到 0）。
# MAGIC Free Edition 可能不支持在线特征存储。学完务必运行 `99_cleanup`。把 `RUN_SERVING` 改成 True 再运行。

# COMMAND ----------

# MAGIC %pip install -U -q "mlflow[databricks]>=3.1" databricks-feature-engineering databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

RUN_SERVING = False

from datetime import timedelta

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
fe = FeatureEngineeringClient()
w = WorkspaceClient()
champion_version = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version
print(f"将部署 {MODEL_NAME} v{champion_version} → 端点 {SERVING_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 在线特征存储
# MAGIC 特征表是 Delta 表，适合批量读取，但单次查询要几秒；在线特征存储是低延迟数据库（Lakebase），毫秒级按主键查询。
# MAGIC `publish_table` 把特征表同步过去（需要特征表开启 Change Data Feed，阶段 3 已开启）。

# COMMAND ----------

if RUN_SERVING:
    store = fe.get_online_store(name=ONLINE_STORE_NAME)
    if store is None:
        # 👀 界面：左侧 Compute → Lakebase（或 Online stores）中出现这个在线存储，创建需要几分钟
        store = fe.create_online_store(name=ONLINE_STORE_NAME, capacity="CU_1")   # CU_1 是最小规格
        print("已提交创建在线存储；如果下一步报“未就绪”，等几分钟后重跑本单元")
    print(store)

    # 👀 界面：Catalog → main → battery_life_lab 下出现 cell_early_features_online
    fe.publish_table(
        online_store=store,
        source_table_name=FEATURE_TABLE,
        online_table_name=ONLINE_FEATURE_TABLE,
        publish_mode="TRIGGERED",   # 手动触发同步；特征表更新后再调用一次即可
    )
    print(f"✅ 已发布 {FEATURE_TABLE} → {ONLINE_FEATURE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 部署 Serving 端点
# MAGIC 端点检测到模型需要查特征，会自动连接在线特征存储。部署需要 10–20 分钟。

# COMMAND ----------

if RUN_SERVING:
    served = [ServedEntityInput(
        name="cell-life-champion",           # 实体名：自己起，端点内唯一
        entity_name=MODEL_NAME,              # UC 中的模型
        entity_version=str(champion_version),
        workload_size="Small",
        scale_to_zero_enabled=True,          # 空闲约 30 分钟后缩到 0，不收算力费；代价是冷启动要几分钟
    )]
    try:
        w.serving_endpoints.get(SERVING_ENDPOINT)
        # 👀 界面：Serving 页面该端点状态变为 Updating，完成后回到 Ready
        w.serving_endpoints.update_config_and_wait(
            name=SERVING_ENDPOINT, served_entities=served, timeout=timedelta(minutes=40))
    except NotFound:
        # 👀 界面：Serving 页面出现 <你的用户名>-cell-life 端点，状态 Not Ready → Ready
        w.serving_endpoints.create_and_wait(
            name=SERVING_ENDPOINT,
            config=EndpointCoreConfigInput(name=SERVING_ENDPOINT, served_entities=served),   # 新版 SDK 要求 config 里也写 name
            timeout=timedelta(minutes=40),
        )
    print("状态:", w.serving_endpoints.get(SERVING_ENDPOINT).state.ready)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 调用：只传电芯 ID

# COMMAND ----------

if RUN_SERVING:
    cell_ids = [r.cell_id for r in spark.table(CELL_META_TABLE).where("dataset_role = 'new_batch'").limit(3).collect()]
    # 👀 界面：端点页面 → Metrics 的请求数增加
    resp = w.serving_endpoints.query(name=SERVING_ENDPOINT, dataframe_records=[{"cell_id": c} for c in cell_ids])
    for cell_id, log_life in zip(cell_ids, resp.predictions):
        print(f"{cell_id}: 预测寿命 ≈ {round(10 ** float(log_life))} 个循环")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✍️ 练习
# MAGIC 1. 用本地终端 + 个人访问令牌（PAT）调用这个端点（REST 写法见文档 9.4）。
# MAGIC 2. 等端点缩到 0 后再调用一次，记录冷启动时间。测试台软件能接受这个等待吗？
# MAGIC 3. 用完立刻运行 `99_cleanup`，删除端点和在线存储。
