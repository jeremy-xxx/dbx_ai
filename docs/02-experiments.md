# 阶段 2：Experiments —— 用 MLflow 追踪每一次训练

> **学习目标**
> 1. 理解 Experiment / Run / Param / Metric / Artifact / Logged Model 的关系；
> 2. 会手动记录（`log_param` / `log_metric` / `log_model`）和自动记录（`autolog`）；
> 3. 会在 UI 中对比多个 Run、画图、筛选最佳模型；
> 4. 知道 Experiments 同样用于 GenAI 的 **Trace** 和 **评估**。
>
> **业务任务**：根据 BMS 遥测数据，预测车辆 **未来 90 天是否发生电池故障**（二分类，正样本约 12%）。
>
> **耗时**：1 小时 **前置**：阶段 0 **费用**：只有 notebook 计算费

---

## 2.1 概念

```
Experiment（实验：一个文件夹，通常对应一个建模问题）  /Users/you/ev-battery-risk
 ├── Run 1  (logreg_balanced)   ← 一次训练
 │    ├── Params:  C=1.0, class_weight=balanced
 │    ├── Metrics: roc_auc=0.90, recall=0.80, pr_auc=0.62
 │    ├── Artifacts: confusion_matrix.png, feature_importance.png
 │    └── Logged Model: model/ （MLflow 3 中模型是“一等公民”，有自己的 ID 和页面）
 ├── Run 2  (random_forest)
 ├── Run 3  (gbdt_grid_search) ── 子 Run（nested）：超参搜索的每个组合
 └── Traces（GenAI 调用链，阶段 1 / 7 生成）
```

| 名词 | 含义 | 类比 |
|---|---|---|
| **Experiment** | 一组相关 Run 的容器 | 一个项目文件夹 |
| **Run** | 一次代码执行（通常是一次训练） | 一次实验记录 |
| **Param** | 输入：超参数、配置 | 实验条件 |
| **Metric** | 输出：可比较的数值，可以按 step 记录曲线 | 实验结果 |
| **Artifact** | 任意文件：图片、CSV、模型 | 附件 |
| **Logged Model** | MLflow 3 新概念：带签名、依赖、输入样例的模型包 | 可部署的“成品” |
| **Trace** | 一次 LLM/Agent 调用的完整调用链 | 分布式追踪 |

> 🔑 **为什么要用？** 没有追踪时，“上周那个效果最好的模型用的什么参数？”往往答不上来。MLflow 让**每一个结果都可复现**。

### 选什么指标？
电池故障是 **少数类（~12%）**，而且 **漏报的代价远大于误报**（漏掉一块要出问题的电池 → 抛锚甚至安全事故；误报 → 多做一次检测）。所以：

| 指标 | 本项目中的角色 |
|---|---|
| `roc_auc` | 整体排序能力，作为 **主排序指标** |
| `pr_auc` | 少数类场景比 ROC AUC 更敏感 |
| `recall` | **漏报率 = 1 - recall**，业务最关心 |
| `precision` | 误报多少，影响售后工作量 |
| `accuracy` | ⚠️ 几乎没用：全猜“不故障”也有 88% |

---

## 2.2 代码实践：`02_experiments_tracking.py`

### Step 1：设置实验
```python
mlflow.set_experiment(EXPERIMENT_PATH)   # 不存在会自动创建
```

### Step 2：自动记录（autolog）
```python
mlflow.sklearn.autolog()
with mlflow.start_run(run_name="autolog_demo_logreg"):
    pipe.fit(X_train, y_train)
```
一行代码自动记录：所有超参、训练指标、模型、输入样例。**适合快速试验**。

### Step 3：手动记录（更可控）
```python
with mlflow.start_run(run_name=name) as run:
    mlflow.log_params(params)
    mlflow.log_metrics({"roc_auc": auc, "pr_auc": ap, "recall": rec, ...})
    mlflow.log_figure(fig, "confusion_matrix.png")
    mlflow.sklearn.log_model(
        sk_model=pipe, name="model",
        signature=signature, input_example=X_train.head(3),
        pyfunc_predict_fn="predict_proba",   # ← 服务时返回概率而不是 0/1
    )
    mlflow.set_tag("candidate", "true")
```
关键点：
- **signature**（签名）：记录输入输出的列名与类型，Serving 时会做校验，**必须要有**（UC 注册模型强制要求）。
- **input_example**：UI 中展示样例，也用于部署前校验。
- **pyfunc_predict_fn="predict_proba"**：通过通用 `pyfunc` 接口（也就是 Serving）预测时返回 `[P(正常), P(故障)]`，后面要用概率划分风险等级。
- **tag**：给 Run 打标签，方便后面筛选。

### Step 4：嵌套 Run 做超参搜索
```python
with mlflow.start_run(run_name="gbdt_grid_search"):
    for lr in [0.05, 0.1]:
        for depth in [2, 3]:
            with mlflow.start_run(run_name=f"gbdt_lr{lr}_d{depth}", nested=True):
                ...
```

### Step 5：特征重要性
notebook 会为随机森林记录一张特征重要性图。看看是不是 `soh_pct`、`cell_voltage_imbalance_mv`、`bms_alerts_90d` 排在前面 —— 模型学到的规律应该 **符合领域常识**，否则要怀疑数据或特征有问题。

### Step 6：用代码查询最佳 Run
```python
best = mlflow.search_runs(
    filter_string="tags.candidate = 'true'",
    order_by=["metrics.roc_auc DESC"], max_results=1)
```
阶段 3 就用这个方法找到要注册的模型。

---

## 2.3 UI 实操（跑完 notebook 后）

1. 左侧 **AI/ML → Experiments** → 找到 `ev-battery-risk`。
2. **表格视图**：
   - 点列设置，把 `roc_auc`、`recall`、`pr_auc`、`model_type` 加到表格里；
   - 在搜索框输入 `metrics.recall > 0.6 and tags.candidate = 'true'` 过滤。
3. **对比**：勾选 3–4 个 Run → **Compare**：
   - **Parallel Coordinates**：看超参数与指标的关系；
   - **Scatter**：x=`precision`，y=`recall`，直观感受二者此消彼长；
4. **Chart 视图**：切换到图表视图，查看各 Run 的指标柱状图。
5. 打开某个 Run → **Artifacts**：查看 `confusion_matrix.png`、`feature_importance.png`、`model/MLmodel`、`requirements.txt`、`input_example.json`。
6. 打开 **Models（Logged Models）** 标签：MLflow 3 中每个 logged model 有独立页面。
7. 切到 `ev-battery-genai` 实验 → **Traces**：这是阶段 1 生成的 LLM 调用记录。

---

## 2.4 练习

1. **阈值调整**：默认 0.5 阈值下 recall 不够高。在 notebook 里计算阈值为 0.2 / 0.3 / 0.4 时的 precision、recall，把结果用 `mlflow.log_metrics` 记到同一个 Run。你会给售后部门推荐哪个阈值？
2. **加一个模型**：在 `CANDIDATES` 里加 `HistGradientBoostingClassifier`，看它排第几。
3. **记录数据集**：`mlflow.log_input(mlflow.data.from_pandas(train_df, name="battery_train"), context="training")`，在 Run 页面查看 **Datasets**。
4. **思考题**：如果去掉 `soh_pct` 这个最强特征，模型还能靠什么预测？（提示：`cycle_count`、`fast_charge_ratio` 间接反映衰减）

## 2.5 检查点

- [ ] Experiments 页面里至少有 4 个 `candidate=true` 的 Run
- [ ] 会用 Compare 功能找出最佳 Run
- [ ] 能说出为什么本任务不看 accuracy
- [ ] 知道 `signature` 为什么是必须的

## 2.6 常见坑

| 现象 | 原因 / 解决 |
|---|---|
| `log_model` 报 `name` 参数不存在 | MLflow < 3 用的是 `artifact_path="model"`；升级 MLflow 或改参数名 |
| 签名里整型列告警 “integer column may have missing values” | 训练前把数值列转成 `float64`（notebook 已处理） |
| 实验创建权限错误 | `EXPERIMENT_PATH` 必须在你有写权限的目录（默认用你的用户目录） |

➡️ 下一步：[阶段 3：Models](03-models.md)
