# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · 全局配置
# MAGIC
# MAGIC 其它 notebook 通过 `%run ./00_config` 引用本文件，所以 **只在这里改配置**。
# MAGIC
# MAGIC 运行本文件会：① 定义全局变量（表名、模型名、端点名…）② 定义 `get_llm_client()` ③ 创建并切换到 schema（已存在则跳过，不删数据）④ 打印当前配置。
# MAGIC **平时不需要单独运行**，其它 notebook 会自动执行它；只在首次导入或修改配置后单独运行一次，用来检查打印结果。详见 `docs/00-environment-setup.md` 0.4.1 节。
# MAGIC
# MAGIC | 变量 | 含义 |
# MAGIC |---|---|
# MAGIC | `CATALOG` / `SCHEMA` | 本项目所有表、模型、函数、Volume 所在位置 |
# MAGIC | `LLM_MODEL` | 大模型名（Unity Gateway 中的 `system.ai.*` 模型服务，需支持 tool calling） |
# MAGIC | `RISK_ENDPOINT` | 阶段 4 部署的电池风险模型端点名 |

# COMMAND ----------

import re

# ======== 需要你修改的配置 ========
CATALOG = "main"        # ← 改成你有 CREATE SCHEMA 权限的 catalog（Free Edition 常见为 "workspace"）
SCHEMA = "ev_battery_lab"

# 基础大模型，通过 Unity Gateway 调用（按 token 计费，需支持 function/tool calling）。
# 可用模型：左侧 AI/ML → Unity Gateway → Models（Location 为 system.ai 的那些），名字填 system.ai.<服务名>
# Free Edition 实测（2026-09）：
#   ✅ system.ai.qwen3-next-80b-a3b-instruct（中文好、支持工具调用、返回纯文本）← 默认
#   ✅ system.ai.llama-4-maverick（备选）
#   ⚠️ 推理类模型（gpt-oss、qwen35-122b-a10b）返回“推理块列表”而非纯文本，不建议选
# 这个名字同时用于 chat 调用、ai_query 和 Agent
LLM_MODEL = "system.ai.qwen3-next-80b-a3b-instruct"
# ==================================

def get_llm_client():
    """返回一个指向 Unity Gateway 的 OpenAI 兼容客户端（地址与鉴权已配置好）。"""
    from databricks.sdk import WorkspaceClient
    from openai import OpenAI
    w = WorkspaceClient()
    token = w.config.authenticate()["Authorization"].split(" ", 1)[1]
    return OpenAI(api_key=token, base_url=f"{w.config.host.rstrip('/')}/ai-gateway/mlflow/v1")

USER = spark.sql("SELECT current_user()").first()[0]
# 端点名在 workspace 内全局唯一，加上用户名前缀避免与同事冲突
_USER_SLUG = re.sub(r"[^a-z0-9]", "-", USER.split("@")[0].lower())[:20].strip("-")

# 表
VEHICLES_TABLE = f"{CATALOG}.{SCHEMA}.vehicles"                    # 车辆档案 + BMS 遥测 + 标签
ACTIONS_TABLE = f"{CATALOG}.{SCHEMA}.maintenance_actions"          # 车型 × 风险等级 → 处置动作
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.battery_features"             # 阶段 6 特征表
POLICY_TABLE = f"{CATALOG}.{SCHEMA}.policy_texts"                  # 规程文档全文

# 模型（Unity Catalog 注册模型）
BASE_MODEL_NAME = f"{CATALOG}.{SCHEMA}.battery_risk_model"         # 阶段 2/3/4
FS_MODEL_NAME = f"{CATALOG}.{SCHEMA}.battery_risk_model_fs"        # 阶段 6（基于特征表训练）
AGENT_MODEL_NAME = f"{CATALOG}.{SCHEMA}.battery_service_agent"     # 阶段 7

# Serving 端点
RISK_ENDPOINT = f"{_USER_SLUG}-battery-risk"

# UC 函数（Agent 工具）
FN_PROFILE = f"{CATALOG}.{SCHEMA}.get_vehicle_profile"
FN_FEATURES = f"{CATALOG}.{SCHEMA}.get_battery_features"
FN_ACTIONS = f"{CATALOG}.{SCHEMA}.get_maintenance_actions"
FN_POLICY = f"{CATALOG}.{SCHEMA}.get_policy_doc"

# Volume（存放规程文档）
POLICY_VOLUME = f"{CATALOG}.{SCHEMA}.policy_docs"
POLICY_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/policy_docs"

# MLflow 实验
EXPERIMENT_PATH = f"/Users/{USER}/ev-battery-risk"
GENAI_EXPERIMENT_PATH = f"/Users/{USER}/ev-battery-genai"

# 模型输入特征（阶段 2 训练、阶段 4 服务、阶段 7 Agent 调用都用这一组）
NUMERIC_FEATURES = [
    "vehicle_age_months", "odometer_km", "cycle_count", "fast_charge_ratio",
    "avg_depth_of_discharge", "max_cell_temp_c_30d", "cell_voltage_imbalance_mv",
    "bms_alerts_90d", "soh_pct",
]
CATEGORICAL_FEATURES = ["vehicle_model", "battery_chemistry", "climate_zone"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LABEL = "battery_fault_90d"

# ---------- 阶段 9：真实数据（MIT/Stanford/TRI 电芯循环寿命数据集） ----------
RAW_VOLUME = f"{CATALOG}.{SCHEMA}.raw_battery_data"
RAW_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw_battery_data"
CELL_META_TABLE = f"{CATALOG}.{SCHEMA}.cell_metadata"          # 每个电芯：批次、充电策略、循环寿命、数据集划分
CELL_CYCLE_TABLE = f"{CATALOG}.{SCHEMA}.cell_cycle_summary"    # 每个电芯每个循环：容量、内阻、温度、充电时间
CELL_CURVE_TABLE = f"{CATALOG}.{SCHEMA}.cell_early_curves"     # 第 10 / 100 循环的放电容量-电压曲线
CELL_FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.cell_early_features" # 前 100 循环提取的特征（特征表）
CELL_PRED_TABLE = f"{CATALOG}.{SCHEMA}.cell_life_predictions"  # 批量预测结果
CELL_MODEL_NAME = f"{CATALOG}.{SCHEMA}.cell_cycle_life_model"
CELL_EXPERIMENT_PATH = f"/Users/{USER}/ev-battery-cell-life"
FN_CELL_LIFE = f"{CATALOG}.{SCHEMA}.get_cell_life_prediction"

# 安全红线（与 battery_maintenance_sop.md 保持一致）
REDLINE_TEMP_C = 60
REDLINE_IMBALANCE_MV = 150

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"""配置已加载
  user            = {USER}
  schema          = {CATALOG}.{SCHEMA}
  LLM_MODEL       = {LLM_MODEL}
  RISK_ENDPOINT   = {RISK_ENDPOINT}
  EXPERIMENT_PATH = {EXPERIMENT_PATH}""")
