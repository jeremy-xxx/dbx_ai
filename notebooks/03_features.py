# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · 特征工程：从前 100 个循环提取特征 → 特征表
# MAGIC
# MAGIC 对应文档：`docs/03-features.md`
# MAGIC
# MAGIC 为什么要放进**特征表**（Feature Store），而不是在训练 notebook 里临时算？
# MAGIC - **训练和预测用同一套特征**：阶段 4 训练、阶段 6 给新电芯打分都从这张表取特征，口径一定一致；
# MAGIC - **可以重复运行**：新一批电芯测完 100 个循环后，重跑本 notebook，新电芯的特征按主键合并进表里；
# MAGIC - **血缘可追溯**：在 Features 页面能看到特征来自哪些表、被哪个模型使用。
# MAGIC
# MAGIC **只用前 100 个循环的数据**。这是业务约束（只测一周就要出结果），也防止用到“未来”的数据。

# COMMAND ----------

# MAGIC %pip install -U -q databricks-feature-engineering

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import numpy as np
import pandas as pd

cycles = spark.table(CELL_CYCLE_TABLE).where(f"cycle BETWEEN 1 AND {EARLY_CYCLES}").toPandas()   # 只读前 100 个循环
curves = spark.table(CELL_CURVE_TABLE).toPandas().set_index("cell_id")
meta = spark.table(CELL_META_TABLE).select("cell_id", "charge_time_0_80_min").toPandas().set_index("cell_id")
print(f"{curves.shape[0]} 个电芯待计算特征")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 提取特征
# MAGIC
# MAGIC | 类别 | 特征 | 物理含义 |
# MAGIC |---|---|---|
# MAGIC | **ΔQ(V) 统计量**（最重要） | `dq_log_var`、`dq_log_abs_min`、`dq_log_abs_mean`、`dq_skew`、`dq_kurt` | 第 100 与第 10 次放电曲线的差异：形状变化越大，衰减机制越活跃 |
# MAGIC | 容量衰减 | `qd_cycle2`、`qd_cycle100`、`qd_max_minus_cycle2`、`fade_slope_2_100`、`fade_intercept_2_100`、`fade_slope_91_100` | 前期容量的水平和变化速度 |
# MAGIC | 充电 / 温度 / 内阻 | `chargetime_avg_first5`、`tmax_max_2_100`、`tmin_min_2_100`、`tavg_sum_2_100`、`ir_cycle2`、`ir_min_2_100`、`ir_diff_100_2` | 发热和内阻增长 |
# MAGIC | 快充策略 | `charge_time_0_80_min` | 测试开始前就已知的设计参数 |
# MAGIC
# MAGIC 缺失值（个别电芯某些循环的测量值异常）**留空不填**，交给阶段 4 的模型流水线处理：
# MAGIC 用训练集的中位数填补，避免测试集和新批次的信息“泄漏”进训练。

# COMMAND ----------

def safe_log10(x: float) -> float:
    return float(np.log10(abs(x))) if x not in (0, None) and np.isfinite(x) else np.nan

feature_rows = []
for cell_id in curves.index:
    s = cycles[cycles.cell_id == cell_id].sort_values("cycle")
    early = s[(s.cycle >= 2) & (s.cycle <= EARLY_CYCLES)]
    ok = early[(early.qd > 0.8) & (early.qd < 1.2)]   # 去掉明显的测量异常点
    if len(ok) < 10:
        print(f"⚠️ {cell_id} 有效循环太少，跳过")
        continue

    dq = pd.Series(np.array(curves.loc[cell_id, "qdlin_cycle100"]) - np.array(curves.loc[cell_id, "qdlin_cycle10"]))
    qd2 = ok.qd.iloc[0]
    slope, intercept = np.polyfit(ok.cycle, ok.qd, 1)
    late = ok[ok.cycle >= EARLY_CYCLES - 9]
    ir = early.ir.where(early.ir > 0)   # 内阻为 0 是测量缺失

    feature_rows.append({
        "cell_id": cell_id,
        # ΔQ(V) 统计量
        "dq_log_var": safe_log10(dq.var()),
        "dq_log_abs_min": safe_log10(dq.min()),
        "dq_log_abs_mean": safe_log10(dq.mean()),
        "dq_skew": float(dq.skew()),
        "dq_kurt": float(dq.kurt()),
        # 容量衰减
        "qd_cycle2": float(qd2),
        "qd_cycle100": float(ok.qd.iloc[-1]),
        "qd_max_minus_cycle2": float(ok.qd.max() - qd2),
        "fade_slope_2_100": float(slope),
        "fade_intercept_2_100": float(intercept),
        "fade_slope_91_100": float(np.polyfit(late.cycle, late.qd, 1)[0]) if len(late) >= 3 else np.nan,
        # 充电、温度、内阻
        "chargetime_avg_first5": float(s[(s.cycle >= 2) & (s.cycle <= 6)].chargetime.mean()),
        "tmax_max_2_100": float(early.tmax.max()),
        "tmin_min_2_100": float(early.tmin.min()),
        "tavg_sum_2_100": float(early.tavg.sum()),
        "ir_cycle2": float(ir.dropna().iloc[0]) if ir.notna().any() else np.nan,
        "ir_min_2_100": float(ir.min()),
        "ir_diff_100_2": float(ir.dropna().iloc[-1] - ir.dropna().iloc[0]) if ir.notna().sum() >= 2 else np.nan,
        # 快充策略
        "charge_time_0_80_min": float(meta.loc[cell_id, "charge_time_0_80_min"]),
    })

features = pd.DataFrame(feature_rows)
features["computed_at"] = pd.Timestamp.now(tz="UTC")
print(f"{len(features)} 个电芯 × {features.shape[1] - 2} 个特征")
print("含缺失值的特征:", features.columns[features.isna().any()].tolist())
display(features.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 写入特征表
# MAGIC 第一次运行时创建；之后再运行，按主键 `cell_id` **合并（merge）**：已有电芯的特征被更新，新电芯被追加。

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()
features_sdf = spark.createDataFrame(features)

if spark.catalog.tableExists(FEATURE_TABLE):
    # 👀 界面：cell_early_features 表 → History 标签多一条 MERGE 记录
    fe.write_table(name=FEATURE_TABLE, df=features_sdf, mode="merge")
    print(f"✅ 已合并进 {FEATURE_TABLE}")
else:
    # 👀 界面：左侧 Features 页面能搜到 cell_early_features；Catalog 里它显示为带主键的表
    fe.create_table(
        name=FEATURE_TABLE,
        primary_keys=["cell_id"],
        df=features_sdf,
        description=f"电芯前 {EARLY_CYCLES} 个循环提取的早期特征（ΔQ(V) 统计量、容量衰减、充电、温度、内阻、快充策略）。数据来源 Severson et al. 2019",
    )
    print(f"✅ 已创建 {FEATURE_TABLE}")

# 开启 Change Data Feed（变更数据流）：可选的阶段 9 把特征表发布到在线特征存储时需要它
spark.sql(f"ALTER TABLE {FEATURE_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# 👀 界面：Features / Catalog 中该表 Columns 标签的 Comment 列显示这些说明
for col, comment in {
    "dq_log_var": "log10(Var(Q100(V)-Q10(V)))：第100与第10次放电曲线差值的方差，论文中最强的单一特征",
    "fade_slope_2_100": "第2-100循环放电容量的线性拟合斜率（Ah/循环）",
    "ir_diff_100_2": "第100循环与第2循环的内阻差（Ω），内阻增长越快通常寿命越短",
    "charge_time_0_80_min": "快充策略对应的0→80%充电时间（分钟），测试开始前已知",
    "computed_at": "特征计算时间（UTC）",
}.items():
    spark.sql(f"ALTER TABLE {FEATURE_TABLE} ALTER COLUMN {col} COMMENT '{comment}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✍️ 练习
# MAGIC 1. 打开左侧 **Features** 页面找到 `cell_early_features`，查看列说明；阶段 4 训练完后再回来看 **Lineage**，能看到哪个模型用了它。
# MAGIC 2. 新增一个特征 `qd_ratio_100_2 = qd_cycle100 / qd_cycle2`，重跑本 notebook，观察表的 **History** 标签多了一条 MERGE。
# MAGIC 3. 思考：为什么 `charge_time_0_80_min` 可以作为特征，而 `cycle_life` 绝对不能？（提示：哪些信息在“测完 100 个循环”的时间点上是已知的？）
