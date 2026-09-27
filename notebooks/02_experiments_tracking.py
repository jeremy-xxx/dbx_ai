# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Experiments：用 MLflow 追踪训练
# MAGIC
# MAGIC 对应文档：`docs/02-experiments.md`
# MAGIC
# MAGIC 任务：根据 BMS 遥测数据预测 **未来 90 天是否发生电池故障**。训练多个模型，把参数、指标、图表、模型全部记录到 MLflow。

# COMMAND ----------

# MAGIC %pip install -U -q "mlflow[databricks]>=3.1" scikit-learn matplotlib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# （界面无变化）只影响本 notebook 之后注册模型的去向（Unity Catalog）
mlflow.set_registry_uri("databricks-uc")
# 👀 界面：左侧 Experiments 页面出现 ev-battery-risk（Workspace → Users → 你的邮箱 下也有）；
#          首次运行时为空，已存在则直接使用、不会清空
exp = mlflow.set_experiment(EXPERIMENT_PATH)
print("experiment_id:", exp.experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 准备数据

# COMMAND ----------

df = spark.table(VEHICLES_TABLE).toPandas()
X = df[MODEL_FEATURES].astype({c: "float64" for c in NUMERIC_FEATURES})  # 整型转 float，避免签名告警
y = df[LABEL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(X_train.shape, X_test.shape, f"故障样本比例 {y.mean():.1%}")

def make_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("pre", pre), ("clf", estimator)])

def evaluate(pipe, threshold: float = 0.5) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "recall": recall_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred),
        "accuracy": accuracy_score(y_test, pred),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. autolog：一行代码自动记录
# MAGIC 运行后打开 Experiments 页面，看看自动记录了哪些东西。

# COMMAND ----------

# 👀 界面：下面这次训练会在 ev-battery-risk 里新增 Run “autolog_demo_logreg”，
#          自动记录的内容在 Run 详情的 Parameters、Metrics、Artifacts 区
mlflow.sklearn.autolog(log_input_examples=True)

with mlflow.start_run(run_name="autolog_demo_logreg") as run:
    pipe = make_pipeline(LogisticRegression(max_iter=2000))
    pipe.fit(X_train, y_train)
    mlflow.log_metrics({f"test_{k}": v for k, v in evaluate(pipe).items()})

mlflow.sklearn.autolog(disable=True)  # 后面改用手动记录
print("run_id:", run.info.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 手动记录：多个候选模型
# MAGIC 每个候选模型一个 Run，记录 params / metrics / 混淆矩阵 / 特征重要性 / 模型，并打上 `candidate=true` 标签。
# MAGIC `class_weight="balanced"` 让模型更重视少数类（故障），通常能提高 recall。

# COMMAND ----------

CANDIDATES = {
    "logreg": (LogisticRegression, {"C": 1.0, "max_iter": 2000}),
    "logreg_balanced": (LogisticRegression, {"C": 1.0, "max_iter": 2000, "class_weight": "balanced"}),
    "random_forest": (RandomForestClassifier, {"n_estimators": 300, "max_depth": 10, "class_weight": "balanced", "random_state": 42}),
    "gbdt": (GradientBoostingClassifier, {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3, "random_state": 42}),
}

def log_feature_importance(pipe):
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(clf.feature_importances_, index=names).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(6, 4))
    imp.plot.barh(ax=ax)
    ax.set_title("Feature importance")
    fig.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)

def train_and_log(run_name: str, estimator_cls, params: dict, nested: bool = False) -> dict:
    # 👀 界面：每调用一次，ev-battery-risk 里多一个 Run（名字 = run_name）
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        pipe = make_pipeline(estimator_cls(**params))
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe)

        mlflow.log_param("model_type", estimator_cls.__name__)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, ax=ax)
        # 👀 界面：Run 详情 → Artifacts 里出现 confusion_matrix.png，点击可预览
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        log_feature_importance(pipe)

        signature = infer_signature(X_train, pipe.predict_proba(X_train.head(5)))
        # 👀 界面：Run 详情 → Artifacts 出现 model/ 目录（MLmodel、requirements.txt、input_example.json）；
        #          实验页面的 Models 标签（Logged Models）也会多一条
        mlflow.sklearn.log_model(
            sk_model=pipe,
            name="model",                        # MLflow < 3 请改为 artifact_path="model"
            signature=signature,
            input_example=X_train.head(3),
            pyfunc_predict_fn="predict_proba",   # Serving 时返回 [P(正常), P(故障)]
            # 新版 MLflow 默认用更安全的 skops 格式保存 sklearn 模型，只允许白名单里的类型；
            # 树模型（随机森林、GBDT）内部的决策树类型不在默认白名单，需要显式信任，否则报 “untrusted types”
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        mlflow.set_tags({"candidate": "true", "stage": "lab-02"})
        print(f"{run_name:<22} auc={metrics['roc_auc']:.4f} recall={metrics['recall']:.3f} "
              f"precision={metrics['precision']:.3f}  run_id={run.info.run_id}")
        return metrics

for name, (cls, params) in CANDIDATES.items():
    train_and_log(name, cls, params)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 嵌套 Run：简单的网格搜索

# COMMAND ----------

# 👀 界面：出现父 Run “gbdt_grid_search”，点它左侧的 ▶ 展开可看到 4 个子 Run
with mlflow.start_run(run_name="gbdt_grid_search"):
    best_auc = 0.0
    for lr in [0.05, 0.1]:
        for depth in [2, 3]:
            m = train_and_log(
                f"gbdt_lr{lr}_d{depth}",
                GradientBoostingClassifier,
                {"n_estimators": 200, "learning_rate": lr, "max_depth": depth, "random_state": 42},
                nested=True,
            )
            best_auc = max(best_auc, m["roc_auc"])
    mlflow.log_metric("best_child_roc_auc", best_auc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 用代码查询最佳 Run

# COMMAND ----------

runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="tags.candidate = 'true'",
    order_by=["metrics.roc_auc DESC"],
)
display(runs[["run_id", "tags.mlflow.runName", "params.model_type",
              "metrics.roc_auc", "metrics.pr_auc", "metrics.recall", "metrics.precision"]])

best = runs.iloc[0]
print(f"🏆 最佳: {best['tags.mlflow.runName']}  auc={best['metrics.roc_auc']:.4f}  run_id={best.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✍️ 练习
# MAGIC 1. 阈值调整：对最佳模型计算 threshold = 0.2 / 0.3 / 0.4 的 precision、recall，记录到 MLflow（见文档 2.4）。
# MAGIC 2. 在 `CANDIDATES` 里加一个 `HistGradientBoostingClassifier`。
# MAGIC 3. 去 UI 用 **Compare → Scatter**（x=precision, y=recall）比较各模型。
