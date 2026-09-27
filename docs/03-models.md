# 阶段 3：Models —— Unity Catalog 模型注册与版本管理

> **学习目标**
> 1. 理解 “Logged Model（实验里的模型）” 与 “Registered Model（注册模型）” 的区别；
> 2. 会把最佳 Run 中保存的**模型**注册到 Unity Catalog，并管理版本、描述、标签（Run 本身仍留在实验里，注册的是它产出的模型）；
> 3. 掌握 **别名（Alias）** 工作流：`champion` / `challenger`；
> 4. 会用别名加载模型，对整个车队做批量风险打分（pandas & Spark）；
> 5. 了解模型的权限与血缘。
>
> **耗时**：45 分钟 **前置**：阶段 2 **费用**：几乎为零

---

## 3.1 概念

```
Experiment 里的 Run ──(register)──►  Registered Model: main.ev_battery_lab.battery_risk_model
   runs:/<id>/model                     ├── Version 1  ← alias: champion
                                        ├── Version 2  ← alias: challenger
                                        └── Version 3
```

| 概念 | 说明 |
|---|---|
| **Registered Model** | UC 中的一个对象，三级命名 `catalog.schema.model`，可以 GRANT 权限 |
| **Model Version** | 每次注册产生一个递增版本号，**不可变** |
| **Alias（别名）** | 指向某个版本的可移动“指针”，如 `@champion`。下游代码只认别名，上线/回滚只需移动别名 |
| **Tags / Description** | 元数据，支持 Markdown 描述 |
| **Lineage（血缘）** | 自动记录：模型 ← 哪个 Run ← 哪些表训练；模型 → 被哪个 Serving 端点使用 |

> ⚠️ 旧版 Workspace Model Registry 的 “Stage（Staging/Production）” 在 UC 中已被 **Alias** 取代。

**模型 URI 速查**

| 写法 | 含义 |
|---|---|
| `runs:/<run_id>/model` | 某次 Run 记录的模型 |
| `models:/main.ev_battery_lab.battery_risk_model/2` | 注册模型第 2 版 |
| `models:/main.ev_battery_lab.battery_risk_model@champion` | 别名当前指向的版本 ✅ 推荐下游使用 |

---

## 3.2 代码实践：`03_models_registry.py`

### Step 1：把 Run 里的模型注册到 UC
```python
mlflow.set_registry_uri("databricks-uc")
mv = mlflow.register_model(f"runs:/{run_id}/model", BASE_MODEL_NAME)
#                           └── 模型地址：这次 Run 里名为 "model" 的模型
```
注意区分：**Run** 是一次训练记录，留在 Experiments 里；**注册**的是 Run 产出的**模型**，它在 UC 中成为 `battery_risk_model` 的一个新版本。一个 Run 里的模型可以被注册多次，每次都产生一个新版本。

### Step 2：补充元数据
```python
client = MlflowClient()
client.update_registered_model(BASE_MODEL_NAME, description="...")
client.update_model_version(BASE_MODEL_NAME, mv.version, description="...")
client.set_model_version_tag(BASE_MODEL_NAME, mv.version, "validation_status", "approved")
```

### Step 3：设置别名
```python
client.set_registered_model_alias(BASE_MODEL_NAME, "champion", mv.version)
```

### Step 4：Champion vs Challenger（带业务闸门）
notebook 先把第二名 Run 的模型注册为 Version 1（模拟“线上已有旧模型”），再把最佳 Run 的模型注册为 Version 2，并设为 `challenger`。
在 **独立评估集**（这里用车龄最大的 1000 辆车模拟“老化严重的车队”）上比较：

- 主指标：ROC AUC 更高；
- **安全闸门**：阈值 0.3 下 recall 不能比 champion 更低（漏报不能变多）。

两条都满足才把 `champion` 别名移给 challenger。这就是最简单的“模型上线闸门”。

### Step 5：用别名加载与推理
```python
model = mlflow.pyfunc.load_model(f"models:/{BASE_MODEL_NAME}@champion")
model.predict(sample_df)            # → [[P(正常), P(故障)], ...]

udf = mlflow.pyfunc.spark_udf(spark, f"models:/{BASE_MODEL_NAME}@champion", result_type="array<double>")
spark.table(VEHICLES_TABLE).withColumn("proba", udf(*MODEL_FEATURES))   # 5000 辆车分布式打分
```

---

## 3.3 UI 实操

1. 左侧 **AI/ML → Models**（或 **Catalog → ev_battery_lab → Models**）→ 打开 `battery_risk_model`。
2. **Overview**：查看各版本、别名、描述。试着在 UI 里：
   - 编辑模型描述（支持 Markdown）；
   - 给某个版本 **Add alias** → `staging`，然后再删除它。
3. 打开某个版本：
   - **Source run** 链接会跳回阶段 2 的 Run；
   - **Lineage** 标签页：上游 `vehicles` 表（通过 Spark 读取时才能自动捕获）。
4. **Permissions** 标签页：了解可授予的权限（`EXECUTE` 用于加载/推理，`MANAGE`/`OWNER` 用于管理）。

对应 SQL：
```sql
-- 授权售后数据组可以加载/推理这个模型（还需要对 catalog/schema 的 USE 权限）
GRANT EXECUTE ON MODEL main.ev_battery_lab.battery_risk_model TO `aftersales-data`;
```

> 注意：本 notebook 每次重跑都会注册新版本，这是正常的——版本号只增不减。

---

## 3.4 练习

1. **回滚演练**：把 `champion` 别名手动指回 Version 1，再用 `@champion` 加载，验证预测变化 —— 感受“上线/回滚只改指针”。
2. **按别名查询**：`client.get_model_version_by_alias(BASE_MODEL_NAME, "champion")`，打印版本号和 run_id。
3. **车队风险报表**：把 Spark UDF 的结果写成 `fleet_risk_scores` 表，按 `vehicle_model` × 风险等级统计车辆数；去 Catalog 查看它与模型之间的血缘。
4. **思考题**：为什么下游（Serving、批处理作业）应该引用别名而不是版本号？

## 3.5 检查点

- [ ] `battery_risk_model` 至少有 2 个版本，`champion` 别名存在
- [ ] 能用 `models:/...@champion` 加载模型并预测
- [ ] 理解 alias 替代 stage 的原因，以及“安全闸门”的意义

## 3.6 常见坑

| 现象 | 原因 / 解决 |
|---|---|
| `Model signature is required` | UC 要求必须有签名，回到阶段 2 确认 `signature=` 已传 |
| `PERMISSION_DENIED: CREATE MODEL` | 需要对 schema 有 `CREATE MODEL` 权限 |
| `register_model` 注册到了旧的 Workspace Registry | 忘了 `mlflow.set_registry_uri("databricks-uc")` 或模型名不是三级命名 |

➡️ 下一步：[阶段 4：Serving](04-serving.md)
