# Databricks notebook source
# MAGIC %md
# MAGIC # 09b · 用前 100 个循环预测电芯寿命（真实数据）
# MAGIC
# MAGIC 对应文档：`docs/09-real-data.md`
# MAGIC
# MAGIC 核心问题：**电芯只循环了 100 次（容量几乎还没开始衰减）时，能否预测它最终能循环多少次？**
# MAGIC 这能把寿命测试从几个月缩短到几天 —— 用于电芯来料检验、快充策略筛选。
# MAGIC
# MAGIC 复用阶段 2/3/6 的平台能力：
# MAGIC ```
# MAGIC 特征提取 → Features（特征表 + FeatureLookup）→ Experiments（3 组特征 × 2 种模型对比）
# MAGIC        → Models（注册 champion）→ 批量预测表 → UC 函数（给 Agent 当工具）
# MAGIC ```
# MAGIC 前置：运行过 `09a_real_data_ingest`。

# COMMAND ----------

# MAGIC %pip install -U -q "mlflow[databricks]>=3.1" databricks-feature-engineering scikit-learn matplotlib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

# （界面无变化）决定下面 register_model 注册到 Unity Catalog
mlflow.set_registry_uri("databricks-uc")
# 👀 界面：左侧 Experiments 页面出现 ev-battery-cell-life（和车队故障模型的实验分开）
exp = mlflow.set_experiment(CELL_EXPERIMENT_PATH)

meta = spark.table(CELL_META_TABLE).toPandas()
cycles = spark.table(CELL_CYCLE_TABLE).where("cycle BETWEEN 1 AND 100").toPandas()
curves = spark.table(CELL_CURVE_TABLE).toPandas().set_index("cell_id")
print(f"{len(meta)} 个电芯", meta.split.value_counts().to_dict())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 直观理解核心特征 ΔQ(V)
# MAGIC `ΔQ(V) = Q₁₀₀(V) − Q₁₀(V)`：同一电压下，第 100 次与第 10 次放电容量之差。
# MAGIC 在前 100 个循环里容量本身几乎没变，但 **放电曲线的形状** 已经悄悄改变 —— 论文发现这条差值曲线的方差（取 log）与寿命高度相关。

# COMMAND ----------

voltage = np.linspace(3.6, 2.0, 1000)  # Qdlin 的电压网格：3.6V → 2.0V，1000 点
sample = meta.sort_values("cycle_life").iloc[:: max(1, len(meta) // 8)]

fig, ax = plt.subplots(figsize=(7, 4))
cmap = plt.cm.viridis
lo, hi = np.log10(meta.cycle_life.min()), np.log10(meta.cycle_life.max())
for _, r in sample.iterrows():
    dq = np.array(curves.loc[r.cell_id, "qdlin_cycle100"]) - np.array(curves.loc[r.cell_id, "qdlin_cycle10"])
    ax.plot(dq, voltage, color=cmap((np.log10(r.cycle_life) - lo) / (hi - lo)), label=f"{r.cycle_life} cycles")
ax.set_xlabel("ΔQ = Q100 - Q10 (Ah)"); ax.set_ylabel("Voltage (V)"); ax.legend(fontsize=7)
ax.set_title("ΔQ(V) curves (brighter = longer life)")  # matplotlib 默认无中文字体，图内文字用英文
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 特征提取（每个电芯一行）
# MAGIC 全部特征只使用 **前 100 个循环** 的数据 —— 这是防止“用未来预测未来”的关键约束。

# COMMAND ----------

def safe_log10(x: float) -> float:
    return float(np.log10(abs(x))) if x not in (0, None) and np.isfinite(x) else np.nan

def extract_features(cell_id: str) -> dict:
    s = cycles[cycles.cell_id == cell_id].sort_values("cycle")
    early = s[(s.cycle >= 2) & (s.cycle <= 100)]
    ok = early[(early.qd > 0.8) & (early.qd < 1.2)]              # 去掉明显的测量异常点
    dq = pd.Series(np.array(curves.loc[cell_id, "qdlin_cycle100"]) - np.array(curves.loc[cell_id, "qdlin_cycle10"]))
    qd2 = ok.loc[ok.cycle == ok.cycle.min(), "qd"].iloc[0]
    slope, intercept = np.polyfit(ok.cycle, ok.qd, 1)
    late = ok[ok.cycle >= 91]
    slope_91 = np.polyfit(late.cycle, late.qd, 1)[0] if len(late) >= 3 else np.nan
    ir = early.ir.where(early.ir > 0)
    return {
        "cell_id": cell_id,
        # ΔQ(V) 统计量（论文最重要的一组特征）
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
        "fade_slope_91_100": float(slope_91),
        # 充电、温度、内阻
        "chargetime_avg_first5": float(s[(s.cycle >= 2) & (s.cycle <= 6)].chargetime.mean()),
        "tmax_max_2_100": float(early.tmax.max()),
        "tmin_min_2_100": float(early.tmin.min()),
        "tavg_sum_2_100": float(early.tavg.sum()),
        "ir_cycle2": float(ir.iloc[0]) if len(ir) else np.nan,
        "ir_min_2_100": float(ir.min()),
        "ir_diff_100_2": float(ir.iloc[-1] - ir.iloc[0]) if len(ir) else np.nan,
    }

feat_pdf = pd.DataFrame([extract_features(c) for c in meta.cell_id])
print("含缺失值的特征:", feat_pdf.columns[feat_pdf.isna().any()].tolist())
feat_pdf = feat_pdf.fillna(feat_pdf.median(numeric_only=True))
display(feat_pdf.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 写入特征表（复习阶段 6）

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()
if spark.catalog.tableExists(CELL_FEATURE_TABLE):
    fe.drop_table(name=CELL_FEATURE_TABLE)
# 👀 界面：左侧 Features 页面能搜到 cell_early_features
fe.create_table(
    name=CELL_FEATURE_TABLE,
    primary_keys=["cell_id"],
    df=spark.createDataFrame(feat_pdf),
    description="电芯前 100 循环提取的早期特征（ΔQ(V) 统计量、容量衰减、充电时间、温度、内阻）。来源 Severson et al. 2019",
)
spark.sql(f"ALTER TABLE {CELL_FEATURE_TABLE} ALTER COLUMN dq_log_var COMMENT 'log10(Var(Q100(V)-Q10(V)))，论文中最强的单一特征'")

labels = spark.createDataFrame(meta.assign(log_cycle_life=np.log10(meta.cycle_life))[
    ["cell_id", "split", "cycle_life", "log_cycle_life"]])
training_set = fe.create_training_set(
    df=labels,
    feature_lookups=[FeatureLookup(table_name=CELL_FEATURE_TABLE, lookup_key="cell_id")],
    label="log_cycle_life",
)
data = training_set.load_df().toPandas()
print(data.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 经典一图：log 方差 ΔQ vs log 寿命

# COMMAND ----------

fig, ax = plt.subplots(figsize=(5, 4))
for split, marker in [("train", "o"), ("test", "s"), ("secondary_test", "^")]:
    d = data[data.split == split]
    ax.scatter(d.dq_log_var, d.cycle_life, label=split, marker=marker, alpha=0.7)
ax.set_yscale("log"); ax.set_xlabel("log10 Var(ΔQ)"); ax.set_ylabel("cycle life"); ax.legend()
corr = np.corrcoef(data.dq_log_var, data.log_cycle_life)[0, 1]
ax.set_title(f"Pearson r = {corr:.2f}")
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Experiments：3 组特征 × 2 种模型
# MAGIC - 特征组参考论文的 variance / discharge / full 三个模型思路（论文建模代码需学术许可，这里是我们自己的实现）；
# MAGIC - 目标是 `log10(cycle_life)`，评估时换回循环次数，指标：**RMSE（循环数）** 与 **MAPE（平均百分比误差）**；
# MAGIC - **样本只有 124 个**：线性模型 + 正则化（ElasticNet）通常比复杂模型更稳，这正是小样本真实数据的常态。

# COMMAND ----------

from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_SETS = {
    "variance": ["dq_log_var"],
    "discharge": ["dq_log_var", "dq_log_abs_min", "dq_skew", "dq_kurt", "qd_cycle2", "qd_max_minus_cycle2"],
    "full": ["dq_log_var", "dq_log_abs_min", "fade_slope_2_100", "fade_intercept_2_100", "qd_cycle2",
             "chargetime_avg_first5", "tavg_sum_2_100", "ir_min_2_100", "ir_diff_100_2"],
}
MODELS = {
    "elasticnet": lambda: Pipeline([("scale", StandardScaler()),
                                    ("reg", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5, max_iter=50000))]),
    "gbdt": lambda: GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05,
                                              subsample=0.8, random_state=42),
}
train = data[data.split == "train"]

def metrics_for(model, feats, split) -> dict:
    d = data[data.split == split]
    pred = 10 ** model.predict(d[feats])
    true = d.cycle_life.values
    return {f"{split}_rmse_cycles": float(np.sqrt(np.mean((pred - true) ** 2))),
            f"{split}_mape_pct": float(np.mean(np.abs(pred - true) / true) * 100)}

results = []
for fs_name, feats in FEATURE_SETS.items():
    for m_name, make in MODELS.items():
        # 👀 界面：ev-battery-cell-life 里共新增 6 个 Run（模型 × 特征组），可用 Compare 对比 test_mape_pct
        with mlflow.start_run(run_name=f"{m_name}_{fs_name}") as run:
            model = make().fit(train[feats], train.log_cycle_life)
            m = {}
            for split in ["train", "test", "secondary_test"]:
                m.update(metrics_for(model, feats, split))
            mlflow.log_params({"feature_set": fs_name, "model": m_name, "n_features": len(feats),
                               "features": ",".join(feats)})
            mlflow.log_metrics(m)

            d = data[data.split != "train"]
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.scatter(d.cycle_life, 10 ** model.predict(d[feats]), c=(d.split == "test"), cmap="coolwarm", alpha=0.7)
            lim = [data.cycle_life.min() * 0.8, data.cycle_life.max() * 1.1]
            ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel("observed"); ax.set_ylabel("predicted"); ax.set_title(f"{m_name} / {fs_name}")
            mlflow.log_figure(fig, "pred_vs_obs.png"); plt.close(fig)

            mlflow.sklearn.log_model(
                sk_model=model, name="model",
                signature=infer_signature(train[feats], model.predict(train[feats].head(3))),
                input_example=train[feats].head(3),
            )
            mlflow.set_tags({"stage": "lab-09", "target": "log10_cycle_life"})
            results.append({"run_id": run.info.run_id, "model": m_name, "feature_set": fs_name, **m})

res = pd.DataFrame(results).sort_values("test_mape_pct")
display(res.round(1))

# COMMAND ----------

# MAGIC %md
# MAGIC 👉 去 **Experiments → ev-battery-cell-life**，用 Compare 看：
# MAGIC - 特征越多一定越好吗？
# MAGIC - `test` 和 `secondary_test` 的误差差距说明了什么？（batch3 是不同时间开始的实验，存在批次差异 → 分布漂移）
# MAGIC - GBDT 的 train 误差很低、test 误差却不低 —— 典型的小样本过拟合。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 注册最佳模型（按 test MAPE 选择；secondary_test 作为上线前的最终检查）

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()
best = res.iloc[0]
# 👀 界面：Models 中出现 cell_cycle_life_model，下一行给最新版本加上 @champion
mv = mlflow.register_model(f"runs:/{best.run_id}/model", CELL_MODEL_NAME)
client.set_registered_model_alias(CELL_MODEL_NAME, "champion", mv.version)
client.update_registered_model(
    CELL_MODEL_NAME,
    description=("## 电芯循环寿命早期预测\n"
                 "- 输入：前 100 循环提取的特征（见 signature）\n"
                 "- 输出：log10(循环寿命)，使用时取 10**y\n"
                 "- 数据：Severson et al., Nature Energy 2019（CC BY 4.0），A123 LFP 18650 电芯，30℃ 快充\n"
                 "- ⚠️ 适用范围：同型号电芯、相近测试条件；不可直接外推到整车电池包"),
)
client.set_model_version_tag(CELL_MODEL_NAME, mv.version, "feature_set", best.feature_set)
print(f"✅ {CELL_MODEL_NAME} v{mv.version} ← {best.model}/{best.feature_set} "
      f"(test MAPE {best.test_mape_pct:.1f}%, secondary MAPE {best.secondary_test_mape_pct:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. 批量预测 → 结果表

# COMMAND ----------

champion = mlflow.pyfunc.load_model(f"models:/{CELL_MODEL_NAME}@champion")
feats = champion.metadata.get_input_schema().input_names()

pred = data[["cell_id", "split", "cycle_life"]].copy()
pred["predicted_cycle_life"] = np.round(10 ** np.asarray(champion.predict(data[feats]))).astype(int)
pred["abs_pct_error"] = (np.abs(pred.predicted_cycle_life - pred.cycle_life) / pred.cycle_life * 100).round(1)
pred = pred.merge(meta[["cell_id", "batch", "charge_policy"]], on="cell_id")
pred["model_version"] = str(mv.version)

# 👀 界面：Tables 出现 cell_life_predictions
(spark.createDataFrame(pred)
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(CELL_PRED_TABLE))
display(spark.table(CELL_PRED_TABLE).orderBy("abs_pct_error", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. UC 函数：让 Agent 也能查电芯寿命预测（可选，见文档 9.7）

# COMMAND ----------

# 👀 界面：Functions 下出现 get_cell_life_prediction
spark.sql(f"""
CREATE OR REPLACE FUNCTION {FN_CELL_LIFE}(
  cid STRING COMMENT '电芯ID，格式如 b1c5（批次1第5号）、b3c10'
)
RETURNS STRING
COMMENT '查询实验室电芯的循环寿命早期预测结果（基于前100循环）：充电策略、预测寿命、实际寿命、误差。数据来自 Severson et al. 2019 公开数据集，仅适用于 A123 LFP 18650 电芯。'
RETURN (
  SELECT to_json(named_struct(
    'cell_id', cell_id, 'charge_policy', charge_policy, 'split', split,
    'predicted_cycle_life', predicted_cycle_life, 'observed_cycle_life', cycle_life,
    'abs_pct_error', abs_pct_error, 'model_version', model_version))
  FROM {CELL_PRED_TABLE} WHERE cell_id = cid LIMIT 1
)
""")
display(spark.sql(f"SELECT {FN_CELL_LIFE}('{pred.cell_id.iloc[0]}') AS result"))
