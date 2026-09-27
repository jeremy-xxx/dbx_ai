# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · 生成合成 BMS 遥测数据
# MAGIC
# MAGIC 创建两张表：
# MAGIC - `vehicles`：5000 辆电动车的档案、电池遥测指标和标签 `battery_fault_90d`（未来 90 天是否发生电池故障）
# MAGIC - `maintenance_actions`：处置动作规则（车型 × 风险等级）
# MAGIC
# MAGIC 数据生成逻辑（符合电池常识）：
# MAGIC 1. 循环次数由里程和电池容量推算；
# MAGIC 2. SOH 衰减随 **循环次数、快充占比、高温、NMC 三元体系、放电深度、车龄** 加快；
# MAGIC 3. 约 3% 的车存在隐性电芯缺陷 → 压差和温度异常；
# MAGIC 4. 故障概率由 SOH、压差、告警、过温、快充等通过 logistic 函数合成。

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 5000

vehicle_model = rng.choice(["city_ev", "suv_ev", "commercial_van"], size=N, p=[0.45, 0.35, 0.20])
capacity_kwh = np.select([vehicle_model == "city_ev", vehicle_model == "suv_ev"], [50.0, 80.0], 70.0)
chemistry = np.where(
    vehicle_model == "suv_ev",
    rng.choice(["NMC", "LFP"], size=N, p=[0.8, 0.2]),
    rng.choice(["LFP", "NMC"], size=N, p=[0.8, 0.2]),
)
climate = rng.choice(["cold", "temperate", "hot"], size=N, p=[0.25, 0.5, 0.25])
owner_type = np.where(vehicle_model == "commercial_van", "fleet",
                      rng.choice(["private", "fleet"], size=N, p=[0.9, 0.1]))

age = rng.integers(1, 97, size=N)                                              # 车龄 1–96 月
daily_km = np.where(vehicle_model == "commercial_van",
                    rng.normal(120, 30, N), rng.normal(45, 15, N)).clip(5, 300)
odometer = (age * 30 * daily_km * rng.uniform(0.85, 1.15, N)).round()
cycles = (odometer / (capacity_kwh * 4.5)).round().astype(int)                 # 约 4.5 km/kWh·循环
fast_ratio = np.where(vehicle_model == "commercial_van", rng.beta(4, 4, N), rng.beta(2, 5, N)).round(3)
dod = rng.uniform(0.3, 0.9, N).round(2)

is_hot = (climate == "hot").astype(float)
is_nmc = (chemistry == "NMC").astype(float)

temp = 38 + 8 * is_hot - 6 * (climate == "cold") + 10 * fast_ratio + rng.normal(0, 3, N)
degradation = (0.012 * cycles * (1 + 0.3 * is_nmc) * (1 + 0.8 * fast_ratio) * (1 + 0.3 * is_hot) * (0.7 + 0.5 * dod)
               + 0.05 * age + rng.normal(0, 1.5, N))
soh = (100 - degradation).clip(60, 100).round(1)

defect = rng.binomial(1, 0.03, N)                                              # 隐性电芯缺陷
imbalance = (rng.gamma(2, 8, N) + (100 - soh) * 1.5 + defect * rng.uniform(100, 200, N)).round(1)
temp = (temp + defect * rng.uniform(5, 20, N)).round(1)
alerts = rng.poisson(0.2 + imbalance / 60 + np.maximum(0, temp - 45) / 5)

z = (-5.6 + 0.12 * (100 - soh) + 0.02 * imbalance + 0.35 * alerts
     + 0.1 * np.maximum(0, temp - 42) + 1.2 * fast_ratio + 0.4 * is_nmc + 0.3 * is_hot)
fault = rng.binomial(1, 1 / (1 + np.exp(-z)))

surnames = ["王", "李", "张", "刘", "陈", "杨", "赵", "黄", "周", "吴"]
given = ["伟", "芳", "娜", "敏", "静", "磊", "洋", "勇", "艳", "杰", "涛", "明"]
fleets = ["顺达物流", "城配速运", "绿行出行", "东方快运", "安捷配送"]
owner_name = [
    (rng.choice(fleets) + "车队") if t == "fleet" else (rng.choice(surnames) + rng.choice(given))
    for t in owner_type
]
delivery_date = pd.Timestamp("2026-09-01") - pd.to_timedelta(age * 30, unit="D")

pdf = pd.DataFrame({
    "vehicle_id": [f"V{i:05d}" for i in range(1, N + 1)],
    "owner_name": owner_name,
    "owner_type": owner_type,
    "vehicle_model": vehicle_model,
    "battery_chemistry": chemistry,
    "battery_capacity_kwh": capacity_kwh,
    "climate_zone": climate,
    "delivery_date": delivery_date.date,
    "vehicle_age_months": age,
    "odometer_km": odometer,
    "cycle_count": cycles,
    "fast_charge_ratio": fast_ratio,
    "avg_depth_of_discharge": dod,
    "max_cell_temp_c_30d": temp,
    "cell_voltage_imbalance_mv": imbalance,
    "bms_alerts_90d": alerts,
    "soh_pct": soh,
    "battery_fault_90d": fault,
})

print(f"故障率: {pdf.battery_fault_90d.mean():.1%}")
print(pdf.groupby("vehicle_model")[["battery_fault_90d", "soh_pct"]].mean().round(3))
redline = (pdf.max_cell_temp_c_30d >= REDLINE_TEMP_C) | (pdf.cell_voltage_imbalance_mv >= REDLINE_IMBALANCE_MV)
print(f"触发安全红线的车辆: {redline.sum()}")
display(pdf.head(10))

# COMMAND ----------

# 👀 界面：Catalog → main → ev_battery_lab → Tables 出现 vehicles 表；点开看 Sample Data（数据）和 Columns（字段）
(spark.createDataFrame(pdf)
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(VEHICLES_TABLE))

# 👀 界面：表的 Overview 显示这段描述；下面几行给列加的 COMMENT 显示在 Columns 标签的 Comment 列
spark.sql(f"COMMENT ON TABLE {VEHICLES_TABLE} IS 'EV battery lab: 合成车辆档案与 BMS 遥测数据，标签 battery_fault_90d'")
spark.sql(f"ALTER TABLE {VEHICLES_TABLE} ALTER COLUMN vehicle_id SET NOT NULL")
spark.sql(f"ALTER TABLE {VEHICLES_TABLE} ALTER COLUMN soh_pct COMMENT '电池健康度 State of Health，新车为 100'")
spark.sql(f"ALTER TABLE {VEHICLES_TABLE} ALTER COLUMN cell_voltage_imbalance_mv COMMENT '电芯最大压差 (mV)，>=150 为安全红线'")
spark.sql(f"ALTER TABLE {VEHICLES_TABLE} ALTER COLUMN max_cell_temp_c_30d COMMENT '近30天电芯最高温度 (℃)，>=60 为安全红线'")
print(f"✅ 写入 {VEHICLES_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 处置动作规则表
# MAGIC Agent 会通过 UC 函数查询这张表，决定给车辆安排什么维保动作。

# COMMAND ----------

actions = pd.DataFrame([
    ("city_ev",        "low",    "P3", "下次常规保养时读取 BMS 日志，无需额外进站", 0.0),
    ("city_ev",        "medium", "P2", "30 天内预约电池健康检测（容量测试 + 电芯均衡）", 2.0),
    ("city_ev",        "high",   "P1", "7 天内进站：电池包全面检测，必要时更换故障模组", 6.0),
    ("suv_ev",         "low",    "P3", "下次常规保养时读取 BMS 日志，无需额外进站", 0.0),
    ("suv_ev",         "medium", "P2", "30 天内预约电池健康检测 + 热管理系统（冷却液/水泵）检查", 3.0),
    ("suv_ev",         "high",   "P1", "7 天内进站：电池包全面检测 + 热管理系统检查，必要时更换模组", 8.0),
    ("commercial_van", "low",    "P3", "远程监控，建议夜间慢充替代部分快充", 0.0),
    ("commercial_van", "medium", "P2", "14 天内检测；检测前安排夜间低速均衡充电", 3.0),
    ("commercial_van", "high",   "P1", "立即调度备用车，48 小时内进站检测，优先模组更换", 8.0),
], columns=["vehicle_model", "risk_level", "priority", "action", "service_hours"])

# 👀 界面：Tables 下出现 maintenance_actions 表（9 行）
(spark.createDataFrame(actions)
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(ACTIONS_TABLE))
print(f"✅ 写入 {ACTIONS_TABLE}")
display(spark.table(ACTIONS_TABLE))
