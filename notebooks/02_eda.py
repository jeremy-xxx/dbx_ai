# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · 探索分析：建模之前先理解数据
# MAGIC
# MAGIC 对应文档：`docs/02-eda.md`
# MAGIC
# MAGIC 要回答四个问题：
# MAGIC 1. 电芯寿命差异有多大？
# MAGIC 2. 快充策略和寿命有什么关系？
# MAGIC 3. **前 100 个循环里容量几乎没变，凭什么能预测寿命？**
# MAGIC 4. 不同批次的电芯有没有系统性差异？（关系到模型上线后会不会“水土不服”）
# MAGIC
# MAGIC > 图表里的文字用英文：Databricks 的 matplotlib 默认没有中文字体，中文会显示成方框。

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

meta = spark.table(CELL_META_TABLE).toPandas()
print(f"{len(meta)} 个电芯；寿命 {meta.cycle_life.min()}–{meta.cycle_life.max()} 个循环，中位数 {int(meta.cycle_life.median())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 寿命差异有多大
# MAGIC 所有电芯是**同一型号、同一温度**，只有快充策略不同，寿命却相差十几倍。

# COMMAND ----------

fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(meta.cycle_life, bins=30)
ax.axvline(SHORT_LIFE_THRESHOLD, color="red", linestyle="--", label=f"short-life threshold ({SHORT_LIFE_THRESHOLD})")
ax.set_xlabel("cycle life"); ax.set_ylabel("cells"); ax.legend()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 容量衰减曲线
# MAGIC 每条线是一个电芯：x = 循环次数，y = 放电容量。注意前 100 个循环（灰色区域）里，几乎所有曲线都挤在一起。

# COMMAND ----------

cyc = spark.sql(f"""
    SELECT s.cell_id, s.cycle, s.qd, m.cycle_life
    FROM {CELL_CYCLE_TABLE} s JOIN {CELL_META_TABLE} m USING (cell_id)
    WHERE s.qd BETWEEN 0.85 AND 1.15 AND s.cycle % 5 = 0
""").toPandas()

fig, ax = plt.subplots(figsize=(8, 4))
norm = plt.Normalize(np.log10(meta.cycle_life.min()), np.log10(meta.cycle_life.max()))
for cell_id, g in cyc.groupby("cell_id"):
    ax.plot(g.cycle, g.qd, color=plt.cm.viridis(norm(np.log10(g.cycle_life.iloc[0]))), linewidth=0.7)
ax.axvspan(0, EARLY_CYCLES, color="grey", alpha=0.2, label=f"first {EARLY_CYCLES} cycles")
ax.axhline(0.88, color="red", linestyle="--", label="end of life (80% = 0.88 Ah)")
ax.set_xlabel("cycle"); ax.set_ylabel("discharge capacity (Ah)"); ax.legend()
ax.set_title("Capacity fade (brighter = longer life)")
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 快充策略 vs 寿命
# MAGIC 直觉：充得越快，电池越伤。看看数据是否支持，以及支持到什么程度。

# COMMAND ----------

fig, ax = plt.subplots(figsize=(6, 4))
for role, marker in [("train", "o"), ("test", "s"), ("new_batch", "^")]:
    d = meta[meta.dataset_role == role]
    ax.scatter(d.charge_time_0_80_min, d.cycle_life, marker=marker, alpha=0.7, label=role)
ax.set_xlabel("charge time 0-80% (min)"); ax.set_ylabel("cycle life"); ax.legend()
corr = meta[["charge_time_0_80_min", "cycle_life"]].corr().iloc[0, 1]
ax.set_title(f"Faster charging vs life (r = {corr:.2f})")
display(fig)

# COMMAND ----------

# 每种策略的平均寿命（同一策略一般测了 2 个电芯）
display(spark.sql(f"""
    SELECT charge_policy, round(avg(charge_time_0_80_min), 1) AS charge_time_min,
           count(*) AS cells, round(avg(cycle_life)) AS avg_life, min(cycle_life) AS min_life, max(cycle_life) AS max_life
    FROM {CELL_META_TABLE}
    GROUP BY charge_policy
    ORDER BY avg_life DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC **结论**：充电时间越短，寿命总体越短，但相关性不算强，同一策略的两个电芯寿命也可能差很多。
# MAGIC 只看快充策略，预测不准单个电芯的寿命，需要看电芯自身的表现。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 前 100 个循环：容量几乎没变
# MAGIC 计算第 100 循环相对第 2 循环的容量变化。如果它能预测寿命，问题就简单了。

# COMMAND ----------

early = spark.sql(f"""
    SELECT cell_id,
           max(CASE WHEN cycle = 2 THEN qd END)   AS qd2,
           max(CASE WHEN cycle = {EARLY_CYCLES} THEN qd END) AS qd100
    FROM {CELL_CYCLE_TABLE} GROUP BY cell_id
""").toPandas().merge(meta[["cell_id", "cycle_life"]], on="cell_id").dropna()
early["fade_pct"] = (early.qd2 - early.qd100) / early.qd2 * 100

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(early.fade_pct, early.cycle_life, alpha=0.6)
ax.set_yscale("log"); ax.set_xlabel("capacity fade cycle 2 -> 100 (%)"); ax.set_ylabel("cycle life")
r = np.corrcoef(early.fade_pct, np.log10(early.cycle_life))[0, 1]
ax.set_title(f"Early capacity fade vs life (r = {r:.2f})")
display(fig)
print(f"前 100 个循环的容量变化中位数只有 {early.fade_pct.median():.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 关键发现：放电曲线的“形状”已经变了
# MAGIC 容量总量几乎没变，但**同一电压下放出的电量分布**已经不同。
# MAGIC 计算 `ΔQ(V) = Q₁₀₀(V) − Q₁₀(V)`：第 100 次与第 10 次放电，在每个电压点上的容量差。
# MAGIC 论文发现：**这条差值曲线的方差（取 log）与寿命高度相关**。

# COMMAND ----------

curves = spark.table(CELL_CURVE_TABLE).toPandas().merge(meta[["cell_id", "cycle_life"]], on="cell_id")
voltage = np.linspace(3.6, 2.0, 1000)   # Qdlin 的电压网格
curves["delta_q"] = [np.array(b) - np.array(a) for a, b in zip(curves.qdlin_cycle10, curves.qdlin_cycle100)]
curves["log_var_dq"] = curves.delta_q.apply(lambda d: np.log10(np.var(d)))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for _, row in curves.sort_values("cycle_life").iloc[::12].iterrows():   # 每隔 12 个取一个，避免太挤
    axes[0].plot(row.delta_q, voltage, color=plt.cm.viridis(norm(np.log10(row.cycle_life))), label=f"{row.cycle_life}")
axes[0].set_xlabel("dQ = Q100 - Q10 (Ah)"); axes[0].set_ylabel("voltage (V)"); axes[0].legend(fontsize=7, title="cycle life")
axes[0].set_title("dQ(V) curves")

axes[1].scatter(curves.log_var_dq, curves.cycle_life, alpha=0.6)
axes[1].set_yscale("log"); axes[1].set_xlabel("log10 Var(dQ)"); axes[1].set_ylabel("cycle life")
r2 = np.corrcoef(curves.log_var_dq, np.log10(curves.cycle_life))[0, 1]
axes[1].set_title(f"log Var(dQ) vs life (r = {r2:.2f})")
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC 对比第 4 节和第 5 节的相关系数：**ΔQ(V) 的方差比容量变化强得多**。这就是第 3 阶段特征工程的核心。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 批次差异：模型上线后的隐患
# MAGIC batch3 比 batch1、2 晚开始近一年，用的快充策略范围也不同。如果它的分布和训练数据不一样，
# MAGIC 模型在新批次上的误差就会变大（这叫**分布漂移**），阶段 7 会验证这一点。

# COMMAND ----------

display(spark.sql(f"""
    SELECT batch, batch_date, dataset_role, count(*) AS cells,
           round(avg(cycle_life)) AS avg_life, round(stddev(cycle_life)) AS std_life,
           round(avg(charge_time_0_80_min), 1) AS avg_charge_time_min
    FROM {CELL_META_TABLE}
    GROUP BY ALL ORDER BY batch, dataset_role
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✍️ 练习
# MAGIC 1. 第 3 节换成 `c1`（第一段充电倍率）作为横轴，相关性比充电时间强还是弱？
# MAGIC 2. 把第 5 节的 `ΔQ` 换成第 10 和第 50 循环之间的差（需要改 `01_ingest` 的 `LATE_IDX`），相关性会下降多少？
# MAGIC 3. 用 SQL 找出“同一策略下寿命差异最大”的两个电芯，看它们的 ΔQ 曲线有什么不同。
