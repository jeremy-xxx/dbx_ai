# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Features：电池特征表、FeatureLookup 与批量推理
# MAGIC
# MAGIC 对应文档：`docs/06-features.md`

# COMMAND ----------

# MAGIC %pip install -U -q databricks-feature-engineering "mlflow[databricks]>=3.1" scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from pyspark.sql import functions as F

# （界面无变化）决定 fe.log_model 注册到 Unity Catalog
mlflow.set_registry_uri("databricks-uc")
# （界面无变化）本 notebook 的训练 Run 记录到阶段 2 创建的 ev-battery-risk
mlflow.set_experiment(EXPERIMENT_PATH)
fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 特征加工
# MAGIC 原始 12 个特征 + 4 个衍生特征。数值列统一转成 double，与阶段 4 端点的签名保持一致。

# COMMAND ----------

DERIVED_FEATURES = ["fast_charge_cycles", "soh_loss_per_year", "thermal_stress", "imbalance_per_soh_loss"]

def compute_features(vehicles_df):
    return (vehicles_df
        .select(
            "vehicle_id",
            *[F.col(c).cast("double").alias(c) for c in NUMERIC_FEATURES],
            *CATEGORICAL_FEATURES,
        )
        .withColumn("fast_charge_cycles", F.round(F.col("cycle_count") * F.col("fast_charge_ratio"), 1))
        .withColumn("soh_loss_per_year",
                    F.round((100 - F.col("soh_pct")) / (F.greatest(F.col("vehicle_age_months"), F.lit(1.0)) / 12), 3))
        .withColumn("thermal_stress", F.greatest(F.col("max_cell_temp_c_30d") - 45, F.lit(0.0)))
        .withColumn("imbalance_per_soh_loss",
                    F.round(F.col("cell_voltage_imbalance_mv") / (100 - F.col("soh_pct") + 1), 3))
        .na.drop(subset=["vehicle_id"]))

features_df = compute_features(spark.table(VEHICLES_TABLE))
display(features_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 创建特征表
# MAGIC 特征表 = UC 中带主键约束的 Delta 表。重复运行时先删除旧表。

# COMMAND ----------

if spark.catalog.tableExists(FEATURE_TABLE):
    fe.drop_table(name=FEATURE_TABLE)

# 👀 界面：左侧 Features 页面能搜到 battery_features；Catalog 里它显示为带主键的表
fe.create_table(
    name=FEATURE_TABLE,
    primary_keys=["vehicle_id"],
    df=features_df,
    description="动力电池故障风险特征：BMS 遥测 + 衍生特征（EV battery lab 阶段 6）",
)

COMMENTS = {
    "fast_charge_cycles": "快充等效循环次数 = cycle_count × fast_charge_ratio",
    "soh_loss_per_year": "年化 SOH 衰减（百分点/年）",
    "thermal_stress": "近30天最高温度超过45℃的部分（℃）",
    "imbalance_per_soh_loss": "压差 / (SOH损失+1)，数值异常高提示单体缺陷而非正常老化",
}
# 👀 界面：Features / Catalog 中该表 Columns 标签的 Comment 列显示这些说明
for col, comment in COMMENTS.items():
    spark.sql(f"ALTER TABLE {FEATURE_TABLE} ALTER COLUMN {col} COMMENT '{comment}'")
print(f"✅ 特征表 {FEATURE_TABLE} 创建完成，去 AI/ML → Features 页面看看")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 标签表 + 组装训练集
# MAGIC 标签表只有 `vehicle_id` 和 `battery_fault_90d`，特征由 `FeatureLookup` 自动 join 进来。

# COMMAND ----------

labels_df = spark.table(VEHICLES_TABLE).select("vehicle_id", LABEL)

training_set = fe.create_training_set(
    df=labels_df,
    feature_lookups=[FeatureLookup(table_name=FEATURE_TABLE, lookup_key="vehicle_id")],
    label=LABEL,
    exclude_columns=["vehicle_id"],
)
train_pdf = training_set.load_df().toPandas()
print(train_pdf.columns.tolist())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 训练并用 `fe.log_model` 注册
# MAGIC 与阶段 2 的区别：模型包里记录了“要从哪个特征表查哪些特征”。

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ALL_NUMERIC = NUMERIC_FEATURES + DERIVED_FEATURES
X = train_pdf[ALL_NUMERIC + CATEGORICAL_FEATURES]
y = train_pdf[LABEL]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

pipe = Pipeline([
    ("pre", ColumnTransformer([
        ("num", StandardScaler(), ALL_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])),
    ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)),
])

# 👀 界面：ev-battery-risk 里新增 Run “gbdt_with_feature_store”，可和阶段 2 的 Run 对比 roc_auc
with mlflow.start_run(run_name="gbdt_with_feature_store") as run:
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {"roc_auc": roc_auc_score(y_test, proba), "recall": recall_score(y_test, proba >= 0.5)}
    mlflow.log_metrics(metrics)
    mlflow.set_tag("stage", "lab-06")

    # 👀 界面：Models 中出现 battery_risk_model_fs；版本页面和 Features 页面的 Lineage 能看到它与 battery_features 的关联
    fe.log_model(
        model=pipe,
        artifact_path="model",           # 新版本若提示改用 name=，按提示修改
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name=FS_MODEL_NAME,
        infer_input_example=True,
        skops_trusted_types=["sklearn.tree._tree.Tree"],   # GBDT 由决策树组成，需显式信任（见 02 notebook 的说明）
    )
print(f"AUC = {metrics['roc_auc']:.4f}（对比阶段 2 的最佳 AUC，衍生特征有帮助吗？）")

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()
latest = max(int(v.version) for v in client.search_model_versions(f"name='{FS_MODEL_NAME}'"))
# 👀 界面：battery_risk_model_fs 最新版本带上 @champion 标签
client.set_registered_model_alias(FS_MODEL_NAME, "champion", latest)
print(f"✅ {FS_MODEL_NAME} Version {latest} → champion")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. `score_batch`：只传车辆 ID，自动查特征

# COMMAND ----------

ids_df = spark.table(VEHICLES_TABLE).select("vehicle_id").limit(20)
preds = fe.score_batch(model_uri=f"models:/{FS_MODEL_NAME}@champion", df=ids_df)
display(preds.select("vehicle_id", "vehicle_model", "soh_pct", "thermal_stress", "soh_loss_per_year", "prediction"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 模拟新的 BMS 数据到达（merge upsert）
# MAGIC 假设车辆 V00001 最近一次回传：温度飙升、压差变大、告警增多 —— 更新特征后，同一个模型的预测随之变化。

# COMMAND ----------

target = "V00001"
target_df = spark.createDataFrame([(target,)], ["vehicle_id"])
model_uri = f"models:/{FS_MODEL_NAME}@champion"

before = fe.score_batch(model_uri=model_uri, df=target_df).select("prediction").first()[0]

new_telemetry = (spark.table(VEHICLES_TABLE).where(F.col("vehicle_id") == target)
    .withColumn("max_cell_temp_c_30d", F.lit(63))
    .withColumn("cell_voltage_imbalance_mv", F.lit(180.0))
    .withColumn("bms_alerts_90d", F.lit(8))
    .withColumn("soh_pct", F.col("soh_pct") - 6))
# 👀 界面：battery_features 表 → History 标签多一条 MERGE 记录（Delta 表的版本历史）
fe.write_table(name=FEATURE_TABLE, df=compute_features(new_telemetry), mode="merge")

after = fe.score_batch(model_uri=model_uri, df=target_df).select("prediction").first()[0]
print(f"{target} 预测：更新前 {before} → 更新后 {after}  (1 = 90 天内可能故障)")
print("注意：这辆车现在也触发了安全红线（≥60℃、≥150mV），阶段 7 的 Agent 应该识别出来。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✍️ 练习
# MAGIC 见文档 6.6：只选部分特征、新增 `km_per_cycle`、多特征表组合。
