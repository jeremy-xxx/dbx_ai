# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · 建模实验：用前 100 个循环预测循环寿命
# MAGIC
# MAGIC 对应文档：`docs/04-train.md`
# MAGIC
# MAGIC ```
# MAGIC 特征表 ──FeatureLookup──► 训练集（train）/ 测试集（test）
# MAGIC                               │
# MAGIC        4 组特征 × 2 种模型 = 8 个 Run，每个 Run 记录：
# MAGIC        参数、5 折交叉验证误差、测试集误差、预测 vs 真实图、模型（带特征表血缘）
# MAGIC ```
# MAGIC
# MAGIC ⚠️ **本 notebook 完全不使用 `new_batch`（batch3）**。它模拟“上线后才到的新电芯”，留给阶段 6、7 检验模型的真实表现。

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
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")
# 👀 界面：左侧 Experiments 页面出现 battery-cycle-life（首次运行时创建）
exp = mlflow.set_experiment(EXPERIMENT_PATH)
fe = FeatureEngineeringClient()

# 标签表：只有电芯 ID 和要预测的目标。目标取 log10(寿命)：寿命跨度 150–2300，取对数后误差更均衡
labels = (spark.table(CELL_META_TABLE)
          .select("cell_id", "dataset_role", F.log10("cycle_life").alias("log_cycle_life")))
train_labels = labels.where("dataset_role = 'train'").drop("dataset_role")
test_labels = labels.where("dataset_role = 'test'").drop("dataset_role")
print("训练集:", train_labels.count(), "个电芯 | 测试集:", test_labels.count(), "个电芯")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 实验设计
# MAGIC
# MAGIC **4 组特征**（由少到多，参考论文的 variance / discharge / full 思路，并加上快充策略）：
# MAGIC
# MAGIC | 特征组 | 特征 | 想验证什么 |
# MAGIC |---|---|---|
# MAGIC | `variance` | 只有 `dq_log_var` | 单个强特征能做到多好 |
# MAGIC | `discharge` | ΔQ(V) 统计量 + 早期容量 | 只看放电行为 |
# MAGIC | `full` | 放电 + 充电 + 温度 + 内阻 | 更多信息是否更好 |
# MAGIC | `full_policy` | `full` + 快充策略的充电时间 | 已知的设计参数有没有帮助 |
# MAGIC
# MAGIC **2 种模型**：
# MAGIC - `elasticnet`：带正则化的线性模型，小样本下通常更稳；
# MAGIC - `gbdt`：梯度提升树，能拟合非线性关系，但训练集只有约 40 个电芯，容易过拟合。
# MAGIC
# MAGIC **3 个误差指标**（都换算回“循环次数”再计算）：
# MAGIC - `cv_mape`：训练集 5 折交叉验证的平均百分比误差，看**稳定性**（`cv_mape_std` 越小越稳）；
# MAGIC - `test_mape`：测试集平均百分比误差，**阶段 5 决定能否上线的主要依据**；
# MAGIC - `test_rmse`：测试集均方根误差（单位：循环次数）。

# COMMAND ----------

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_SETS = {
    "variance": ["dq_log_var"],
    "discharge": ["dq_log_var", "dq_log_abs_min", "dq_skew", "dq_kurt", "qd_cycle2", "qd_max_minus_cycle2"],
    "full": ["dq_log_var", "dq_log_abs_min", "fade_slope_2_100", "fade_intercept_2_100", "qd_cycle2",
             "chargetime_avg_first5", "tavg_sum_2_100", "ir_min_2_100", "ir_diff_100_2"],
}
FEATURE_SETS["full_policy"] = FEATURE_SETS["full"] + ["charge_time_0_80_min"]

def make_model(model_name: str) -> Pipeline:
    # 缺失值用“训练集”的中位数填补：填补规则只从训练数据学，不会泄漏测试集信息
    if model_name == "elasticnet":
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("reg", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5, max_iter=50000))])
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("reg", GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05,
                                                       subsample=0.8, random_state=42))])

def mape(true_life, pred_life) -> float:
    return float(np.mean(np.abs(pred_life - true_life) / true_life) * 100)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 跑 8 个实验
# MAGIC 每组特征用 `FeatureLookup` 从特征表取数：训练集和测试集走同一条取数路径，口径一致。
# MAGIC 模型用 `fe.log_model` 保存：它会记住“这个模型需要从哪张特征表取哪些特征”，
# MAGIC 所以阶段 6 预测时**只传电芯 ID**，特征自动查出来。

# COMMAND ----------

results = []
for fs_name, feats in FEATURE_SETS.items():
    lookups = [FeatureLookup(table_name=FEATURE_TABLE, lookup_key="cell_id", feature_names=feats)]
    train_set = fe.create_training_set(df=train_labels, feature_lookups=lookups,
                                       label="log_cycle_life", exclude_columns=["cell_id"])
    train_pdf = train_set.load_df().toPandas()
    test_pdf = fe.create_training_set(df=test_labels, feature_lookups=lookups,
                                      label="log_cycle_life", exclude_columns=["cell_id"]).load_df().toPandas()
    X_train, y_train = train_pdf[feats], train_pdf.log_cycle_life
    X_test, y_test = test_pdf[feats], test_pdf.log_cycle_life

    for model_name in ["elasticnet", "gbdt"]:
        # 👀 界面：battery-cycle-life 里每循环一次多一个 Run（名字 = 模型_特征组）
        with mlflow.start_run(run_name=f"{model_name}_{fs_name}") as run:
            # 5 折交叉验证：只在训练集内部轮流留出一折做验证，评估模型是否稳定
            cv_scores = []
            for fit_idx, val_idx in KFold(n_splits=5, shuffle=True, random_state=42).split(X_train):
                m = make_model(model_name).fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
                cv_scores.append(mape(10 ** y_train.iloc[val_idx], 10 ** m.predict(X_train.iloc[val_idx])))

            # 用全部训练集重新训练，在测试集上评估
            model = make_model(model_name).fit(X_train, y_train)
            pred_life, true_life = 10 ** model.predict(X_test), 10 ** y_test
            metrics = {
                "cv_mape": float(np.mean(cv_scores)),
                "cv_mape_std": float(np.std(cv_scores)),
                "train_mape": mape(10 ** y_train, 10 ** model.predict(X_train)),
                "test_mape": mape(true_life, pred_life),
                "test_rmse": float(np.sqrt(np.mean((pred_life - true_life) ** 2))),
            }
            mlflow.log_params({"feature_set": fs_name, "model": model_name, "n_features": len(feats),
                               "features": ",".join(feats), "early_cycles": EARLY_CYCLES})
            mlflow.log_metrics(metrics)

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.scatter(true_life, pred_life, alpha=0.7)
            lim = [100, 2600]
            ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel("observed cycle life"); ax.set_ylabel("predicted cycle life")
            ax.set_title(f"{model_name} / {fs_name}  test MAPE {metrics['test_mape']:.1f}%")
            # 👀 界面：Run 详情 → Artifacts 里出现 pred_vs_obs.png
            mlflow.log_figure(fig, "pred_vs_obs.png"); plt.close(fig)

            # 👀 界面：Run 详情 → Artifacts 出现 model/ 目录；之后在 Features 页面的 Lineage 能看到模型与特征表的关联
            fe.log_model(
                model=model,
                artifact_path="model",
                flavor=mlflow.sklearn,
                training_set=train_set,
                infer_input_example=True,
                # 新版 MLflow 默认用更安全的 skops 格式保存 sklearn 模型，只接受白名单里的类型：
                # numpy.dtype 来自缺失值填补（SimpleImputer），sklearn.tree._tree.Tree 来自 GBDT 的决策树
                skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
            )
            mlflow.set_tags({"stage": "train", "feature_set": fs_name, "model": model_name})
            results.append({"run_id": run.info.run_id, "model": model_name, "feature_set": fs_name, **metrics})
            print(f"{model_name:<10} {fs_name:<12} cv={metrics['cv_mape']:5.1f}%±{metrics['cv_mape_std']:4.1f}  "
                  f"train={metrics['train_mape']:5.1f}%  test={metrics['test_mape']:5.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 结果对比

# COMMAND ----------

import pandas as pd

summary = pd.DataFrame(results).sort_values("test_mape")
display(summary.round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC **怎么看这张表**：
# MAGIC - `train_mape` 远小于 `test_mape`：**过拟合**。GBDT 在约 40 个样本上很容易出现这种情况；
# MAGIC - `cv_mape_std` 大：模型对“训练数据里恰好有哪些电芯”很敏感，不够稳定；
# MAGIC - 特征越多不一定越好：小样本下，多出来的特征可能只是在拟合噪声；
# MAGIC - 论文报告的最佳测试误差约 9%。我们的特征和模型是按论文思路自己实现的，结果在同一量级就正常。
# MAGIC
# MAGIC 👉 去 **Experiments → battery-cycle-life**，勾选几个 Run → **Compare**，用散点图对比 `cv_mape` 和 `test_mape`。

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✍️ 练习
# MAGIC 1. 给 GBDT 换一组更保守的参数（如 `max_depth=1`、`n_estimators=100`），过拟合是否减轻？
# MAGIC 2. 新增一组特征 `ir_only`（只用内阻相关特征），看看单靠内阻能预测到什么程度。
# MAGIC 3. 把 `EARLY_CYCLES` 改成 50（需要重跑 01、03），误差会增加多少？这就是“测试时间 vs 预测精度”的权衡。
