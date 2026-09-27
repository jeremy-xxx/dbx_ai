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

# 读取整张表，转成 pandas DataFrame（5000 行，内存放得下）
df = spark.table(VEHICLES_TABLE).toPandas()

# X = 特征（模型的输入），y = 标签（要预测的答案）
X = df[MODEL_FEATURES]
X = X.astype({col: "float64" for col in NUMERIC_FEATURES})  # 整数列转成小数，避免 MLflow 签名告警
y = df[LABEL]

# 75% 训练、25% 测试；stratify=y 保证两边的故障比例一致；random_state 固定随机种子，每次切分结果相同
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print("训练集:", X_train.shape, " 测试集:", X_test.shape, f" 故障样本比例 {y.mean():.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. autolog：一行代码自动记录
# MAGIC 运行后打开 Experiments 页面，看看自动记录了哪些东西。
# MAGIC
# MAGIC 模型是一条 **Pipeline（流水线）**，数据依次经过两步：
# MAGIC ```
# MAGIC "pre"（预处理）：数值列 → StandardScaler 标准化；类别列 → OneHotEncoder 独热编码
# MAGIC "clf"（分类器）：逻辑回归
# MAGIC ```

# COMMAND ----------

# 👀 界面：下面这次训练会在 ev-battery-risk 里新增 Run “autolog_demo_logreg”，
#          自动记录的内容在 Run 详情的 Parameters、Metrics、Artifacts 区
mlflow.sklearn.autolog(log_input_examples=True)

with mlflow.start_run(run_name="autolog_demo_logreg") as run:
    # 第 1 步：组装流水线
    pipe = Pipeline([
        ("pre", ColumnTransformer([
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ])),
        ("clf", LogisticRegression(max_iter=2000)),
    ])

    # 第 2 步：训练 —— autolog 在这一步自动记录参数、训练集指标、模型
    pipe.fit(X_train, y_train)

    # 第 3 步：autolog 只记录训练集指标，测试集指标要自己算、自己记
    proba = pipe.predict_proba(X_test)[:, 1]      # 每辆车“会故障”的概率
    pred = (proba >= 0.5).astype(int)             # 概率 ≥ 0.5 判为会故障（1），否则 0
    mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, proba))
    mlflow.log_metric("test_recall", recall_score(y_test, pred))
    mlflow.log_metric("test_precision", precision_score(y_test, pred, zero_division=0))

mlflow.sklearn.autolog(disable=True)  # 关闭自动记录，后面改用手动记录
print("run_id:", run.info.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 手动记录：多个候选模型
# MAGIC 每个候选模型一个 Run，记录 params / metrics / 混淆矩阵 / 特征重要性 / 模型，并打上 `candidate=true` 标签。
# MAGIC `class_weight="balanced"` 让模型更重视少数类（故障），通常能提高 recall。

# COMMAND ----------

# 4 个候选模型：名字 → 模型对象（参数直接写在括号里）
CANDIDATES = {
    "logreg":          LogisticRegression(C=1.0, max_iter=2000),
    "logreg_balanced": LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
    "random_forest":   RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42),
    "gbdt":            GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42),
}

for run_name, model in CANDIDATES.items():
    # 👀 界面：每循环一次，ev-battery-risk 里多一个 Run（名字 = run_name）
    with mlflow.start_run(run_name=run_name) as run:

        # 第 1 步：组装流水线（预处理 + 当前这个模型）
        pipe = Pipeline([
            ("pre", ColumnTransformer([
                ("num", StandardScaler(), NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ])),
            ("clf", model),
        ])

        # 第 2 步：训练
        pipe.fit(X_train, y_train)

        # 第 3 步：在测试集上预测
        proba = pipe.predict_proba(X_test)[:, 1]   # 故障概率
        pred = (proba >= 0.5).astype(int)          # 0/1 判定

        # 第 4 步：计算指标
        metrics = {
            "roc_auc":   roc_auc_score(y_test, proba),
            "pr_auc":    average_precision_score(y_test, proba),
            "recall":    recall_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "f1":        f1_score(y_test, pred),
            "accuracy":  accuracy_score(y_test, pred),
        }

        # 第 5 步：记录参数和指标
        # 👀 界面：Run 详情 → Parameters / Metrics
        mlflow.log_param("model_type", type(model).__name__)   # 例如 "LogisticRegression"
        mlflow.log_params(model.get_params())                   # 模型的全部超参数
        mlflow.log_metrics(metrics)

        # 第 6 步：画混淆矩阵并保存为图片
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
        # 👀 界面：Run 详情 → Artifacts 里出现 confusion_matrix.png，点击可预览
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # 第 7 步：树模型才有特征重要性（随机森林、GBDT），逻辑回归跳过
        if hasattr(model, "feature_importances_"):
            feature_names = pipe.named_steps["pre"].get_feature_names_out()   # 独热编码后的列名
            importance = pd.Series(model.feature_importances_, index=feature_names)
            importance = importance.sort_values().tail(12)                    # 取最重要的 12 个
            fig, ax = plt.subplots(figsize=(6, 4))
            importance.plot.barh(ax=ax)
            ax.set_title("Feature importance")
            fig.tight_layout()
            # 👀 界面：Artifacts 里出现 feature_importance.png
            mlflow.log_figure(fig, "feature_importance.png")
            plt.close(fig)

        # 第 8 步：保存模型
        signature = infer_signature(X_train, pipe.predict_proba(X_train.head(5)))   # 记录输入输出的列名和类型
        # 👀 界面：Run 详情 → Artifacts 出现 model/ 目录（MLmodel、requirements.txt、input_example.json）；
        #          实验页面的 Models 标签（Logged Models）也会多一条
        mlflow.sklearn.log_model(
            sk_model=pipe,
            name="model",                        # MLflow < 3 请改为 artifact_path="model"
            signature=signature,
            input_example=X_train.head(3),
            pyfunc_predict_fn="predict_proba",   # 部署后调用时返回 [P(正常), P(故障)]，而不是 0/1
        )

        # 第 9 步：打标签，阶段 3 会按 candidate=true 筛选候选模型
        mlflow.set_tag("candidate", "true")
        mlflow.set_tag("stage", "lab-02")

        print(f"{run_name:<16} auc={metrics['roc_auc']:.4f}  recall={metrics['recall']:.3f}  "
              f"precision={metrics['precision']:.3f}  run_id={run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 嵌套 Run：简单的网格搜索
# MAGIC 尝试 GBDT 的 2 个学习率 × 2 个树深度 = 4 种组合。
# MAGIC 一个 **父 Run** 下面挂 4 个 **子 Run**（`nested=True`），界面上可以折叠/展开。
# MAGIC 这里只关心“哪组超参数更好”，所以子 Run 只记录参数和指标，不保存模型。

# COMMAND ----------

# 👀 界面：出现父 Run “gbdt_grid_search”，点它左侧的 ▶ 展开可看到 4 个子 Run
with mlflow.start_run(run_name="gbdt_grid_search"):
    best_auc = 0.0

    for learning_rate in [0.05, 0.1]:
        for max_depth in [2, 3]:
            child_name = f"gbdt_lr{learning_rate}_d{max_depth}"

            with mlflow.start_run(run_name=child_name, nested=True):   # nested=True → 成为子 Run
                pipe = Pipeline([
                    ("pre", ColumnTransformer([
                        ("num", StandardScaler(), NUMERIC_FEATURES),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
                    ])),
                    ("clf", GradientBoostingClassifier(
                        n_estimators=200, learning_rate=learning_rate, max_depth=max_depth, random_state=42)),
                ])
                pipe.fit(X_train, y_train)

                proba = pipe.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, proba)
                recall = recall_score(y_test, (proba >= 0.5).astype(int))

                mlflow.log_param("learning_rate", learning_rate)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_metric("roc_auc", auc)
                mlflow.log_metric("recall", recall)
                print(f"{child_name:<18} auc={auc:.4f}  recall={recall:.3f}")

            if auc > best_auc:
                best_auc = auc

    # 父 Run 只记录汇总结果
    mlflow.log_metric("best_child_roc_auc", best_auc)
    print(f"最佳子 Run 的 AUC: {best_auc:.4f}")

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
