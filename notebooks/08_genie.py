# Databricks notebook source
# MAGIC %md
# MAGIC # 08 · Genie 数据助手：让工程师用自然语言查电芯数据
# MAGIC
# MAGIC 对应文档：`docs/08-genie.md`
# MAGIC
# MAGIC **业务场景**：电池研发工程师想知道“哪种快充策略寿命最长？”“这批有几个短寿命电芯？”，
# MAGIC 不想每次都找数据团队写 SQL。**Genie Agent** 读懂表结构和业务说明后，把问题自动翻译成 SQL 并给出答案。
# MAGIC
# MAGIC Genie 答得准不准，**主要取决于数据准备得好不好**。本 notebook 做三件事：
# MAGIC 1. 建一张**宽表视图** `cell_overview`：把元数据、关键特征、最新预测合在一起，Genie 不用自己猜怎么关联表；
# MAGIC 2. 给每一列写清楚**业务含义**：Genie 会读取这些说明；
# MAGIC 3. 先用 SQL 算出几个示例问题的**标准答案**，之后用来检查 Genie 答得对不对。
# MAGIC
# MAGIC 前置：运行过 `06_batch_score`（预测表已存在）。

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

assert spark.catalog.tableExists(PREDICTION_TABLE), "预测表不存在，请先运行 06_batch_score"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 建宽表视图
# MAGIC 视图里的列注释写在列名后面（`列名 COMMENT '说明'`），Genie 和 Catalog 都能看到。

# COMMAND ----------

# 👀 界面：Catalog → main → battery_life_lab 下出现视图 cell_overview（图标与表不同），Columns 标签显示每列说明
spark.sql(f"""
CREATE OR REPLACE VIEW {GENIE_VIEW} (
  cell_id               COMMENT '电芯编号，格式 b<批次>c<序号>，例如 b1c5 表示批次1第5号电芯',
  batch                 COMMENT '测试批次：1、2、3（按实验开始日期划分）',
  batch_date            COMMENT '批次的实验开始日期',
  dataset_role          COMMENT 'train=训练集，test=测试集，new_batch=模型上线后新到的一批电芯（batch3）',
  charge_policy         COMMENT '快充策略，格式 C1(Q1%)-C2：先以 C1 倍率充到 Q1% 电量，再以 C2 倍率充到 80%',
  first_step_c_rate     COMMENT '第一段充电倍率（C），越大充得越快',
  switch_soc_pct        COMMENT '两段充电的切换点（电量百分比）',
  second_step_c_rate    COMMENT '第二段充电倍率（C）',
  charge_time_0_80_min  COMMENT '按快充策略计算的 0→80% 充电时间（分钟），越短表示快充越激进',
  actual_cycle_life     COMMENT '实际循环寿命：容量衰减到额定值 80%（0.88 Ah）时的循环次数',
  dq_log_var            COMMENT '前100循环放电曲线变化量的对数方差，越大通常寿命越短，是最重要的预测特征',
  capacity_cycle100_ah  COMMENT '第100循环的放电容量（Ah），额定容量 1.1 Ah',
  resistance_increase   COMMENT '第100与第2循环的内阻差（Ω），越大表示内阻增长越快',
  max_temp_c            COMMENT '前100循环中的最高电芯温度（℃）',
  predicted_cycle_life  COMMENT '模型预测的循环寿命（基于前100循环）',
  qc_decision           COMMENT '质检决策：short_life=预测寿命低于550，建议剔除或复检；borderline=预测范围下限低于550，建议延长测试；normal=正常放行',
  prediction_error_pct  COMMENT '预测误差百分比 = |预测寿命 - 实际寿命| / 实际寿命 × 100',
  model_version         COMMENT '做出该预测的模型版本'
)
COMMENT '电芯总览：一行一个电芯，包含快充策略、实际寿命、关键早期特征和最新一次寿命预测。数据来源 Severson et al. 2019 公开数据集（A123 LFP 18650 电芯，30℃ 快充测试）'
AS
SELECT
  m.cell_id, m.batch, m.batch_date, m.dataset_role, m.charge_policy,
  m.c1, m.q1, m.c2, m.charge_time_0_80_min, m.cycle_life,
  f.dq_log_var, f.qd_cycle100, f.ir_diff_100_2, f.tmax_max_2_100,
  p.predicted_cycle_life, p.qc_decision,
  round(abs(p.predicted_cycle_life - m.cycle_life) / m.cycle_life * 100, 1),
  p.model_version
FROM {CELL_META_TABLE} m
LEFT JOIN {FEATURE_TABLE} f USING (cell_id)
LEFT JOIN (
  SELECT * FROM {PREDICTION_TABLE}
  QUALIFY row_number() OVER (PARTITION BY cell_id ORDER BY scored_at DESC) = 1   -- 每个电芯只取最新一次预测
) p USING (cell_id)
""")
display(spark.table(GENIE_VIEW).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 示例问题的标准答案
# MAGIC 这几条 SQL 有两个用途：
# MAGIC - 在 Genie Agent 里作为 **示例 SQL（Example SQL queries）**，教它怎么写这类查询；
# MAGIC - 作为**标准答案**：在 Genie 里问同样的问题，对比结果是否一致。

# COMMAND ----------

EXAMPLES = {
    "哪种快充策略的平均寿命最长？列出前 5 名": f"""
        SELECT charge_policy, round(avg(charge_time_0_80_min), 1) AS charge_time_min,
               count(*) AS cells, round(avg(actual_cycle_life)) AS avg_cycle_life
        FROM {GENIE_VIEW} GROUP BY charge_policy ORDER BY avg_cycle_life DESC LIMIT 5""",
    "新批次里有多少个电芯需要复检或延长测试？": f"""
        SELECT qc_decision, count(*) AS cells
        FROM {GENIE_VIEW} WHERE dataset_role = 'new_batch' GROUP BY qc_decision""",
    "充电时间少于 10 分钟的电芯，平均寿命比其它电芯短多少？": f"""
        SELECT CASE WHEN charge_time_0_80_min < 10 THEN 'under_10_min' ELSE '10_min_or_more' END AS charge_group,
               count(*) AS cells, round(avg(actual_cycle_life)) AS avg_cycle_life
        FROM {GENIE_VIEW} GROUP BY 1""",
    "新批次里预测误差最大的 3 个电芯是哪些？": f"""
        SELECT cell_id, charge_policy, actual_cycle_life, predicted_cycle_life, prediction_error_pct
        FROM {GENIE_VIEW} WHERE dataset_role = 'new_batch'
        ORDER BY prediction_error_pct DESC LIMIT 3""",
}

for question, sql in EXAMPLES.items():
    print(f"❓ {question}")
    display(spark.sql(sql))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 去界面创建 Genie Agent
# MAGIC 按文档 `docs/08-genie.md` 第 8.3 节操作，概要如下：
# MAGIC
# MAGIC 1. 左侧 **Genie Agents → New**，数据源选 `main.battery_life_lab.cell_overview`（可再加 `cell_cycle_summary` 用于逐循环的问题）→ **Create**；
# MAGIC 2. 创建后 **Genie Code** 会自动读取数据、建议表说明和示例问题，看一遍，接受合适的；
# MAGIC 3. **Instructions** 里粘贴下一个单元格打印的业务说明；
# MAGIC 4. **Example SQL queries** 里添加上面 1–2 条示例（问题 + SQL）；
# MAGIC 5. 用上面 4 个问题和练习中的问题测试，对照标准答案。

# COMMAND ----------

print(f"""电池研发数据助手的业务说明（复制到 Genie Agent 的 Instructions）：

- 本数据是实验室电芯循环寿命测试数据：同型号 A123 LFP 18650 电芯（额定 1.1 Ah），30℃ 恒温，不同快充策略。
- “寿命”指循环寿命：容量衰减到 0.88 Ah（额定 80%）时的循环次数，对应 actual_cycle_life 列。
- “短寿命电芯”指实际寿命低于 {SHORT_LIFE_THRESHOLD} 个循环；“预测为短寿命”看 qc_decision = 'short_life'。
- “需要复检”包括 qc_decision 为 short_life 或 borderline 的电芯。
- “新批次”指 dataset_role = 'new_batch'。问“模型效果”时，只看 new_batch 和 test，不要用 train。
- “快充越激进”指 charge_time_0_80_min 越短。
- 预测误差用 prediction_error_pct 的平均值表示，单位是百分比。
- 回答使用简体中文，数值保留整数或一位小数。""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✍️ 练习
# MAGIC 1. 在 Genie 里问：“第一段充电倍率超过 6C 的电芯，有多少是短寿命？” 对照你自己写的 SQL。
# MAGIC 2. 删掉 Instructions 里关于“短寿命”的定义，再问同样的问题，看 Genie 怎么理解“短寿命”。这说明了业务说明的作用。
# MAGIC 3. 问一个视图里没有数据的问题（如“这些电芯的生产厂家是哪家？”），看 Genie 会不会编造。
