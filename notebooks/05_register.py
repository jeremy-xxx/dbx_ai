# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · 上线闸门：选出候选模型 → 检查是否达标 → 注册为 champion
# MAGIC
# MAGIC 对应文档：`docs/05-register.md`
# MAGIC
# MAGIC ```
# MAGIC 8 个实验 Run ──① 按交叉验证误差选候选──► ② 闸门检查 ──全部通过──► ③ 注册到 Unity Catalog，设为 @champion
# MAGIC                                               │
# MAGIC                                               └── 任何一项不通过 → 不上线，打印原因
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -U -q "mlflow[databricks]>=3.1" databricks-feature-engineering

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

# （界面无变化）决定下面 register_model 注册到 Unity Catalog
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
exp = mlflow.set_experiment(EXPERIMENT_PATH)

runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], filter_string="tags.stage = 'train'")
assert len(runs) > 0, "没有找到训练记录，请先运行 04_train"
display(runs[["tags.mlflow.runName", "metrics.cv_mape", "metrics.cv_mape_std",
              "metrics.train_mape", "metrics.test_mape"]].sort_values("metrics.cv_mape").round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ① 选候选：按交叉验证误差，而不是测试集误差
# MAGIC 如果直接挑“测试集误差最小”的模型，测试集就被用来做选择了，它报告的误差会偏乐观。
# MAGIC 正确做法：**只用训练集信息（交叉验证）选模型，测试集只用来做最后的检查**。

# COMMAND ----------

candidate = runs.sort_values("metrics.cv_mape").iloc[0]
c = {k: float(candidate[f"metrics.{k}"]) for k in ["cv_mape", "cv_mape_std", "train_mape", "test_mape"]}
print(f"候选模型: {candidate['tags.mlflow.runName']}  "
      f"cv={c['cv_mape']:.1f}%±{c['cv_mape_std']:.1f}  train={c['train_mape']:.1f}%  test={c['test_mape']:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ② 闸门检查
# MAGIC
# MAGIC | 检查项 | 规则 | 防止什么 |
# MAGIC |---|---|---|
# MAGIC | 测试集误差达标 | `test_mape ≤ MAX_ACCEPTABLE_MAPE`（默认 15%） | 精度不满足业务要求 |
# MAGIC | 交叉验证稳定 | `cv_mape_std ≤ 0.5 × cv_mape` | 结果靠运气：换几个训练电芯误差就大变 |
# MAGIC | 测试集与交叉验证一致 | `test_mape ≤ 2 × cv_mape` | 测试集表现和训练时的估计差太多，说明有问题 |
# MAGIC | 比现有 champion 更好 | 如果已有 champion，`test_mape` 必须更低 | 新模型反而更差 |
# MAGIC
# MAGIC 这些阈值是业务和团队约定的，可以在 `00_config` 或下面的代码里调整。

# COMMAND ----------

checks = {
    f"测试集误差 ≤ {MAX_ACCEPTABLE_MAPE}%": c["test_mape"] <= MAX_ACCEPTABLE_MAPE,
    "交叉验证稳定（std ≤ 0.5 × 均值）": c["cv_mape_std"] <= 0.5 * c["cv_mape"],
    "测试集与交叉验证一致（test ≤ 2 × cv）": c["test_mape"] <= 2 * c["cv_mape"],
}

try:
    current = client.get_model_version_by_alias(MODEL_NAME, "champion")
    current_mape = float(current.tags.get("test_mape", "inf"))
    checks[f"优于现有 champion（v{current.version}，test {current_mape:.1f}%）"] = c["test_mape"] < current_mape
except MlflowException:
    print("还没有 champion，这是第一次上线")

for name, passed in checks.items():
    print(("✅ " if passed else "❌ ") + name)
PASSED = all(checks.values())

# COMMAND ----------

# MAGIC %md
# MAGIC ## ③ 注册并设为 champion
# MAGIC 注册的是候选 Run 里保存的**模型**（`runs:/<run_id>/model`，最后的 `model` 是阶段 4 `fe.log_model` 时起的名字）。
# MAGIC 版本上记录关键指标作为标签，阶段 6 用 `test_mape` 给预测加误差范围，阶段 7 用 `features` 做漂移检查。

# COMMAND ----------

FORCE_REGISTER = False   # 学习时如果闸门没过、又想继续后面的阶段，可以临时改成 True

if PASSED or FORCE_REGISTER:
    # 👀 界面：左侧 Models（或 Catalog → main → battery_life_lab → Models）出现 cell_cycle_life_model，新增一个版本
    mv = mlflow.register_model(f"runs:/{candidate.run_id}/model", MODEL_NAME)
    for key in ["cv_mape", "cv_mape_std", "test_mape"]:
        client.set_model_version_tag(MODEL_NAME, mv.version, key, f"{c[key]:.2f}")
    client.set_model_version_tag(MODEL_NAME, mv.version, "feature_set", candidate["params.feature_set"])
    client.set_model_version_tag(MODEL_NAME, mv.version, "features", candidate["params.features"])
    client.set_model_version_tag(MODEL_NAME, mv.version, "gate_passed", str(PASSED))
    # 👀 界面：模型页面该版本旁显示 @champion 标签（之前的 champion 自动失去这个标签）
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    # 👀 界面：模型页面 Overview 显示这段描述
    client.update_registered_model(
        MODEL_NAME,
        description=(
            "## 电芯循环寿命早期预测\n"
            f"- 输入：电芯 ID（特征自动从 `{FEATURE_TABLE}` 查询，只用前 {EARLY_CYCLES} 个循环的数据）\n"
            "- 输出：log10(循环寿命)，使用时取 10**y\n"
            "- 数据：Severson et al., Nature Energy 2019（CC BY 4.0），A123 LFP 18650 电芯，30℃ 快充测试\n"
            "- ⚠️ 适用范围：同型号电芯、相近测试条件；不可外推到其它型号或整车电池包"
        ),
    )
    print(f"✅ {MODEL_NAME} v{mv.version} 已设为 champion（闸门{'通过' if PASSED else '未通过，强制注册'}）")
else:
    print("⛔ 闸门未通过，不注册。可以回到 04 调整特征或模型后重跑；学习时也可以把 FORCE_REGISTER 改成 True")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✍️ 练习
# MAGIC 1. 把 `MAX_ACCEPTABLE_MAPE` 改成 8，看候选模型能不能过闸门。真实项目里，这个数应该由谁来定？
# MAGIC 2. 改成按 `test_mape` 选候选，和按 `cv_mape` 选的结果一样吗？如果不一样，说明了什么？
# MAGIC 3. 在 Models 页面打开新版本，点 **Source run** 跳回训练记录，查看它的 `pred_vs_obs.png`。
