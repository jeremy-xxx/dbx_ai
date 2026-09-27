# Databricks notebook source
# MAGIC %md
# MAGIC # 07a · Agent 工具：Unity Catalog 函数 + 维保/质保文档
# MAGIC
# MAGIC 对应文档：`docs/07-agents.md` 第 7.3 节
# MAGIC
# MAGIC Agent = 大模型 + 工具 + 编排逻辑。本 notebook 准备“工具”和“知识”：
# MAGIC
# MAGIC | UC 函数 | 作用 |
# MAGIC |---|---|
# MAGIC | `get_vehicle_profile(vid)` | 车辆档案 + 最新电池遥测 + **是否在质保内** + **是否触发安全红线**（确定性计算，不交给 LLM） |
# MAGIC | `get_battery_features(vid)` | 风险模型所需的 12 个特征（给风险端点用） |
# MAGIC | `get_maintenance_actions(model_name, risk)` | 按车型和风险等级查处置动作 |
# MAGIC | `get_policy_doc(doc_name)` | 返回维保规程 / 质保政策 / 充电 FAQ 全文 |
# MAGIC
# MAGIC 同时把 `data/policies/*.md` 上传到 Volume，供 Agent Bricks 的 Knowledge Assistant 使用。
# MAGIC
# MAGIC 前置：阶段 6 的特征表 `battery_features` 已创建。

# COMMAND ----------

# MAGIC %pip install -U -q "unitycatalog-ai[databricks]" databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 为什么用 UC 函数做工具？
# MAGIC - **治理**：函数在 UC 里，有权限（`EXECUTE`）、有血缘、有审计；
# MAGIC - **复用**：同一个函数，SQL 分析师能用，Playground 能挂，Agent 代码能调；
# MAGIC - **自描述**：函数和参数的 `COMMENT` 会自动变成大模型看到的“工具说明” —— **注释写得好，Agent 才会用对**；
# MAGIC - **确定性**：质保判断、安全红线这类 **规则明确、不能出错** 的逻辑，放在 SQL 里算，而不是让 LLM 去做算术。

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {FN_PROFILE}(
  vid STRING COMMENT '车辆ID，格式为大写字母V加5位数字，例如 V00042'
)
RETURNS STRING
COMMENT '根据车辆ID查询车辆档案与最新电池状态：车主、车型、电芯体系、车龄、里程、SOH、最高温度、压差、BMS告警次数，以及是否在质保期内(in_warranty)、是否触发安全红线(safety_redline)。返回JSON字符串，找不到时返回NULL。'
RETURN (
  SELECT to_json(named_struct(
    'vehicle_id', v.vehicle_id, 'owner_name', v.owner_name, 'owner_type', v.owner_type,
    'vehicle_model', v.vehicle_model, 'battery_chemistry', v.battery_chemistry,
    'climate_zone', v.climate_zone, 'delivery_date', v.delivery_date,
    'vehicle_age_months', v.vehicle_age_months, 'odometer_km', v.odometer_km,
    'soh_pct', f.soh_pct, 'max_cell_temp_c_30d', f.max_cell_temp_c_30d,
    'cell_voltage_imbalance_mv', f.cell_voltage_imbalance_mv, 'bms_alerts_90d', f.bms_alerts_90d,
    'fast_charge_ratio', f.fast_charge_ratio,
    'in_warranty', CASE WHEN v.vehicle_model = 'commercial_van'
                        THEN v.vehicle_age_months < 60 AND v.odometer_km < 300000
                        ELSE v.vehicle_age_months < 96 AND v.odometer_km < 150000 END,
    'soh_below_warranty_threshold', f.soh_pct < 70,
    'safety_redline', f.max_cell_temp_c_30d >= {REDLINE_TEMP_C} OR f.cell_voltage_imbalance_mv >= {REDLINE_IMBALANCE_MV}))
  FROM {VEHICLES_TABLE} v JOIN {FEATURE_TABLE} f ON v.vehicle_id = f.vehicle_id
  WHERE v.vehicle_id = vid
  LIMIT 1
)
""")

feature_struct = ", ".join(f"'{c}', {c}" for c in MODEL_FEATURES)
spark.sql(f"""
CREATE OR REPLACE FUNCTION {FN_FEATURES}(
  vid STRING COMMENT '车辆ID，例如 V00042'
)
RETURNS STRING
COMMENT '返回电池故障风险模型所需的特征（JSON），供内部调用风险模型使用。'
RETURN (
  SELECT to_json(named_struct({feature_struct}))
  FROM {FEATURE_TABLE}
  WHERE vehicle_id = vid
  LIMIT 1
)
""")

spark.sql(f"""
CREATE OR REPLACE FUNCTION {FN_ACTIONS}(
  model_name STRING COMMENT '车型：city_ev / suv_ev / commercial_van',
  risk STRING COMMENT '风险等级：low / medium / high'
)
RETURNS STRING
COMMENT '根据车型和电池故障风险等级，查询标准处置动作（优先级 priority、动作 action、预计工时 service_hours），返回JSON数组。'
RETURN (
  SELECT to_json(collect_list(named_struct('priority', priority, 'action', action, 'service_hours', service_hours)))
  FROM {ACTIONS_TABLE}
  WHERE vehicle_model = lower(model_name) AND risk_level = lower(risk)
)
""")
print("✅ 3 个查询函数已创建")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 规程文档：写入表 + 上传 Volume
# MAGIC - 表 `policy_texts` + 函数 `get_policy_doc()`：给代码版 Agent 用（简单的“知识工具”）
# MAGIC - Volume `policy_docs`：给 Agent Bricks Knowledge Assistant 用（真正的 RAG）

# COMMAND ----------

import os
import shutil
from pathlib import Path

policy_dir = (Path(os.getcwd()) / ".." / "data" / "policies").resolve()
files = sorted(policy_dir.glob("*.md"))
assert files, f"没找到规程文档，请确认 {policy_dir} 存在（见 docs/00 的导入说明）"

rows = [(f.stem, f.read_text(encoding="utf-8")) for f in files]
spark.createDataFrame(rows, ["doc_name", "content"]).write.mode("overwrite").saveAsTable(POLICY_TABLE)

spark.sql(f"CREATE VOLUME IF NOT EXISTS {POLICY_VOLUME} COMMENT '电池售后规程文档（Knowledge Assistant 数据源）'")
for f in files:
    shutil.copy(f, f"{POLICY_VOLUME_PATH}/{f.name}")
print("✅ 已上传:", [f.name for f in files], "→", POLICY_VOLUME_PATH)

spark.sql(f"""
CREATE OR REPLACE FUNCTION {FN_POLICY}(
  doc STRING COMMENT '文档名：battery_maintenance_sop（维保规程：风险等级、安全红线、沟通要求）/ battery_warranty_policy（质保政策：期限、范围、审批）/ charging_faq（充电与续航常见问题）'
)
RETURNS STRING
COMMENT '返回公司电池售后规程文档全文。给出处置建议前必须阅读 battery_maintenance_sop；涉及维修费用或更换时必须阅读 battery_warranty_policy。'
RETURN (SELECT first(content) FROM {POLICY_TABLE} WHERE doc_name = lower(doc))
""")
print("✅ get_policy_doc 已创建")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 用 SQL 测试函数
# MAGIC 找一辆触发安全红线的车和一辆健康的车，看看函数返回什么。

# COMMAND ----------

redline_vid = spark.sql(f"""
  SELECT vehicle_id FROM {FEATURE_TABLE}
  WHERE max_cell_temp_c_30d >= {REDLINE_TEMP_C} OR cell_voltage_imbalance_mv >= {REDLINE_IMBALANCE_MV}
  ORDER BY cell_voltage_imbalance_mv DESC LIMIT 1
""").first()[0]
print("红线样例车辆:", redline_vid)

display(spark.sql(f"""
SELECT {FN_PROFILE}('{redline_vid}')                      AS profile,
       {FN_FEATURES}('{redline_vid}')                     AS features,
       {FN_ACTIONS}('commercial_van', 'high')             AS actions,
       left({FN_POLICY}('battery_warranty_policy'), 80)   AS warranty_head
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 用 Python（unitycatalog-ai）测试 —— Agent 运行时就是这样调用的

# COMMAND ----------

from unitycatalog.ai.core.databricks import DatabricksFunctionClient

uc = DatabricksFunctionClient()
res = uc.execute_function(FN_PROFILE, {"vid": "V00042"})
print("value:", res.value, "\nerror:", res.error)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 👉 去 Playground 试试！
# MAGIC 1. 打开 **AI/ML → Playground**，选择 `LLM_MODEL` 对应的模型；
# MAGIC 2. **Tools → Add tool → Unity Catalog function**，添加 `get_vehicle_profile`、`get_maintenance_actions`、`get_policy_doc`；
# MAGIC 3. System prompt 填：`你是电池售后助手。回答前先用工具查询车辆档案和公司规程，用中文回答。`
# MAGIC 4. 问（把 ID 换成上面打印的红线车辆）：`车辆 V0xxxx 现在什么状况？在质保内吗？应该怎么处理？`
# MAGIC 5. 展开每一步的工具调用，看看模型传了什么参数、是否注意到 `safety_redline=true`；
# MAGIC 6. 点击顶部 **Get code → Create agent notebook** → 会生成一套 Agent notebook 代码，和我们手写的 `agent.py` 对比一下！
# MAGIC    （同一菜单里的 **Export to Databricks Apps** 会导出为 Databricks App，是官方推荐的新部署方式，了解即可。）

# COMMAND ----------

print("本 notebook 创建的函数：")
for fn in [FN_PROFILE, FN_FEATURES, FN_ACTIONS, FN_POLICY]:
    print("  ", fn)
