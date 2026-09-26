# 阶段 6：Features —— 电池特征表与特征工程

> **学习目标**
> 1. 理解为什么需要“特征存储”：**复用、一致性（训练/推理不偏差）、血缘、发现**；
> 2. 在 Unity Catalog 中创建电池特征表（带主键的 Delta 表）；
> 3. 用 `FeatureLookup` + `create_training_set` 组装训练集；
> 4. 用 `fe.log_model` 打包模型，使其 **推理时只需传车辆 ID**，自动查特征；
> 5. 用 `score_batch` 做批量推理；了解时间点查找（BMS 快照场景）与在线特征服务。
>
> **耗时**：1 小时 **前置**：阶段 2 **费用**：只有 notebook 计算费

---

## 6.1 为什么需要特征表？

电池场景里常见的问题：

| 问题 | 例子 |
|---|---|
| **重复开发** | 售后、质量、车联网 3 个团队各算一遍“快充循环次数”“年化 SOH 衰减率” |
| **训练-推理偏差（skew）** | 训练时用 Spark SQL 算特征，车端/平台用 C++/Java 重写，口径不一致 |
| **不可追溯** | 模型用了哪些特征？特征从哪个遥测表来？ |

Databricks 的方案：**UC 中任何带主键的 Delta 表都可以当特征表**。

```
vehicles(遥测) ──特征加工──► battery_features (PK: vehicle_id)
                                    │  FeatureLookup(lookup_key="vehicle_id")
labels(vehicle_id, fault) ──────────┤
                                    ▼
                           create_training_set → 训练 → fe.log_model
                                                          │（模型里记住了“要查哪些特征”）
               推理时只传 vehicle_id ──► score_batch ─────┘ 自动 join 特征
```

## 6.2 本阶段的衍生特征（领域知识 → 特征）

| 衍生特征 | 公式 | 电池领域含义 |
|---|---|---|
| `fast_charge_cycles` | `cycle_count × fast_charge_ratio` | 快充循环的绝对数量，比“占比”更能反映累计伤害 |
| `soh_loss_per_year` | `(100 - soh_pct) / (vehicle_age_months / 12)` | 年化衰减速度，衰减**速度**异常比当前 SOH 更早暴露问题 |
| `thermal_stress` | `max(max_cell_temp_c_30d - 45, 0)` | 超过 45℃ 的热应力，高温加速副反应 |
| `imbalance_per_soh_loss` | `cell_voltage_imbalance_mv / (100 - soh_pct + 1)` | 压差相对衰减程度是否异常 → 可能是单体缺陷而非正常老化 |

> 💡 好特征往往来自领域知识。跑完后对比阶段 2 的 AUC，看看这些特征有没有帮助。

## 6.3 核心 API 速查

```python
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
fe = FeatureEngineeringClient()

# 创建特征表
fe.create_table(name=FEATURE_TABLE, primary_keys=["vehicle_id"], df=features_df,
                description="动力电池特征")
# 更新特征（merge upsert），例如每天新的 BMS 数据到来
fe.write_table(name=FEATURE_TABLE, df=new_features_df, mode="merge")

# 组装训练集
training_set = fe.create_training_set(
    df=labels_df,                                  # 只有 vehicle_id + battery_fault_90d
    feature_lookups=[FeatureLookup(table_name=FEATURE_TABLE, lookup_key="vehicle_id")],
    label="battery_fault_90d",
    exclude_columns=["vehicle_id"],
)
train_pdf = training_set.load_df().toPandas()

# 带特征元数据地记录模型
fe.log_model(model=pipe, artifact_path="model", flavor=mlflow.sklearn,
             training_set=training_set, registered_model_name=FS_MODEL_NAME)

# 批量推理：只给主键
fe.score_batch(model_uri=f"models:/{FS_MODEL_NAME}@champion", df=ids_df)
```

## 6.4 代码实践：`06_feature_engineering.py`

| 步骤 | 内容 |
|---|---|
| 1 | 从 `vehicles` 计算特征：原始 12 个 + 4 个衍生特征 |
| 2 | `create_table` 创建 `battery_features` 特征表，并给列加注释 |
| 3 | 单独构造标签表（真实场景标签往往来自售后工单系统） |
| 4 | `create_training_set` + 训练 + `fe.log_model` 注册 `battery_risk_model_fs` |
| 5 | `score_batch` 只用 `vehicle_id` 批量打分 |
| 6 | **模拟新的 BMS 数据**：某车突然过温、压差飙升，`write_table(mode="merge")` 后重新打分，风险随之变化 |

> 📌 阶段 4 的端点仍然服务 “直接传特征” 的 `battery_risk_model`。`battery_risk_model_fs` 若要做实时 Serving，需要把特征表发布到 **在线特征存储（Online Feature Store）**，端点就能“只传车辆 ID 自动查特征”。这属于进阶内容，见 6.7。

## 6.5 UI 实操

1. 左侧 **AI/ML → Features**：搜索 `battery_features`。
2. 打开特征表：
   - **Overview**：主键、描述、特征列及注释；
   - **Lineage**：上游 `vehicles` 表、下游 `battery_risk_model_fs` 模型；
   - **Producers / Consumers**：哪些 notebook/job 写入、哪些模型使用；
3. 打开 **Models → battery_risk_model_fs** 某个版本，确认能看到它依赖的特征表。
4. 想象你是质量部门的新同事：只看 Features 页面的描述和列注释，能否理解每个特征？不能的话补上 `COMMENT`。

## 6.6 练习

1. **只选部分特征**：`FeatureLookup(..., feature_names=["soh_pct", "soh_loss_per_year", "thermal_stress", "bms_alerts_90d"])`，训练一个“精简版”模型，和全量特征对比 AUC。
2. **新增特征**：加入 `km_per_cycle = odometer_km / (cycle_count + 1)`，用 `write_table(mode="merge")` 写入，重新训练。
3. **多特征表**：新建 `chemistry_climate_features`（主键 `battery_chemistry` + `climate_zone`，如“该组合的平均 SOH 衰减率”），用两个 `FeatureLookup` 组合。
4. **思考题**：如果直接用“该组合的历史故障率”作为特征，会不会造成 **标签泄漏**？如何避免？

## 6.7 进阶：时间点查找与在线服务（了解即可）

- **Time series feature table（非常适合 BMS 遥测）**：真实系统里 BMS 每天都会产生快照。创建特征表时指定 `timeseries_columns="snapshot_date"`，标签表带“故障发生前的观察时间”，`FeatureLookup` 会自动做 **point-in-time join** —— 只取观察时间之前最近的快照，避免用到“故障发生后才出现的高温数据”造成泄漏。
- **Online Feature Store**：把特征表同步到低延迟在线存储，Serving 端点部署 `battery_risk_model_fs` 后只需传 `{"vehicle_id": "V00042"}`。具体步骤随产品迭代变化较快，请参考官方文档 “Online feature stores”。

## 6.8 检查点

- [ ] Features 页面能搜到 `battery_features`，并看到下游模型
- [ ] `score_batch` 只传 `vehicle_id` 就能输出预测
- [ ] 完成第 6 步：更新特征后同一辆车的预测发生了变化
- [ ] 能解释 training-serving skew 以及 point-in-time join 解决的问题

## 6.9 常见坑

| 现象 | 原因 / 解决 |
|---|---|
| `primary key column must be NOT NULL` | 主键列需非空：`df.na.drop(subset=["vehicle_id"])`，必要时 `ALTER COLUMN ... SET NOT NULL` |
| `Table already exists` | `create_table` 只用一次，之后用 `write_table`；或先 `fe.drop_table` |
| `ModuleNotFoundError: databricks.feature_engineering` | `%pip install databricks-feature-engineering`（ML Runtime 自带） |
| `fe.log_model` 不认识 `artifact_path` / `name` | 不同版本参数名不同，按报错提示改名 |
| 衍生特征出现 `inf` / `null` | 除数为 0（新车车龄、SOH=100），notebook 已用 `greatest(...,1)` 等处理 |

➡️ 下一步：[阶段 7：Agents](07-agents.md)
