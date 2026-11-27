# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · 全局配置
# MAGIC
# MAGIC 其它 notebook 通过 `%run ./00_config` 引用本文件，所以 **只在这里改配置**。
# MAGIC
# MAGIC 运行本文件会：① 定义全局变量（表名、模型名、实验路径…）② 创建并切换到 schema（已存在则跳过，不删数据）③ 打印当前配置。
# MAGIC **平时不需要单独运行**，其它 notebook 会自动执行它；只在首次导入或修改配置后单独运行一次，检查打印结果。

# COMMAND ----------

import re

# ======== 需要你修改的配置 ========
CATALOG = "main"              # ← 改成你有 CREATE SCHEMA 权限的 catalog（Free Edition 常见为 "workspace"）
SCHEMA = "battery_life_lab"
# ==================================

USER = spark.sql("SELECT current_user()").first()[0]
# 端点名在 workspace 内全局唯一，加上用户名前缀避免与同事冲突
_USER_SLUG = re.sub(r"[^a-z0-9]", "-", USER.split("@")[0].lower())[:20].strip("-")

# ---------- 原始数据 ----------
RAW_VOLUME = f"{CATALOG}.{SCHEMA}.raw_battery_data"
RAW_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw_battery_data"

# ---------- Delta 表 ----------
CELL_META_TABLE = f"{CATALOG}.{SCHEMA}.cell_metadata"          # 每个电芯一行：批次、充电策略、循环寿命、数据集角色
CELL_CYCLE_TABLE = f"{CATALOG}.{SCHEMA}.cell_cycle_summary"    # 每个电芯每个循环一行：容量、内阻、温度、充电时间
CELL_CURVE_TABLE = f"{CATALOG}.{SCHEMA}.cell_early_curves"     # 第 10 / 100 循环的放电容量-电压曲线
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.cell_early_features"      # 特征表：前 100 个循环提取的特征
PREDICTION_TABLE = f"{CATALOG}.{SCHEMA}.cell_life_predictions" # 批量预测结果
MONITORING_TABLE = f"{CATALOG}.{SCHEMA}.model_monitoring"      # 每次监控的误差与漂移指标
GENIE_VIEW = f"{CATALOG}.{SCHEMA}.cell_overview"               # 给 Genie 用的宽表视图

# ---------- MLflow ----------
EXPERIMENT_PATH = f"/Users/{USER}/battery-cycle-life"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.cell_cycle_life_model"

# ---------- 可选：实时 Serving（阶段 9） ----------
SERVING_ENDPOINT = f"{_USER_SLUG}-cell-life"
ONLINE_STORE_NAME = f"{_USER_SLUG}-cell-features"                  # 在线特征存储（Lakebase）
ONLINE_FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.cell_early_features_online"

# ---------- 业务规则 ----------
EARLY_CYCLES = 100                 # 只用前 100 个循环的数据做预测（约一周的测试时间）
SHORT_LIFE_THRESHOLD = 550         # 预测寿命低于 550 个循环视为“短寿命”电芯（论文中的分类阈值）
MAX_ACCEPTABLE_MAPE = 15.0         # 模型上线的误差要求：测试集平均百分比误差不超过 15%
DRIFT_ALERT_RATIO = 1.5            # 新批次误差超过测试集误差的 1.5 倍时告警

# 👀 界面：Catalog → {CATALOG} 下出现 battery_life_lab 库（已存在则跳过，不删数据）
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"""配置已加载
  user            = {USER}
  schema          = {CATALOG}.{SCHEMA}
  EXPERIMENT_PATH = {EXPERIMENT_PATH}
  MODEL_NAME      = {MODEL_NAME}""")
