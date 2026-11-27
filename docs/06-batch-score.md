# 阶段 6：批量预测

> **目标**：新一批电芯测完 100 个循环后，自动预测寿命、给出质检决策，并配置成定时任务。
> **notebook**：`06_batch_score.py` **耗时**：约 30 分钟

---

## 6.1 为什么是批量预测
实验室是**按批次**测试电芯的：一批电芯同时上测试台，同时测完 100 个循环。所以最自然的用法是一批测完后统一打分，而不是每个电芯单独实时调用。批量预测还有两个好处：不需要常驻的服务，不会持续计费；结果直接写进表里，方便追溯和分析。

（如果确实需要实时调用，见可选的 [阶段 9](09-serving.md)。）

## 6.2 输出：预测 + 误差范围 + 质检决策

| 列 | 含义 |
|---|---|
| `predicted_cycle_life` | 预测寿命 |
| `predicted_low` / `predicted_high` | 误差范围：预测值 × (1 ∓ 模型的测试集平均误差) |
| `qc_decision` | 质检决策（见下表） |
| `model_version`、`scored_at` | 哪个模型版本、什么时候打的分，方便追溯 |

| 决策 | 条件 | 后续动作 |
|---|---|---|
| `short_life` | 预测寿命 < 550 | 剔除或复检 |
| `borderline` | 预测寿命 ≥ 550，但范围下限 < 550 | 延长测试，再观察 |
| `normal` | 范围下限也 ≥ 550 | 正常放行 |

**为什么要有 `borderline`？** 模型有误差。预测寿命是 600 时，真实寿命可能只有 520。把“范围下限跌破阈值”的电芯单独标出来，可以减少漏检：宁可多做一些检测，也不放过短寿命电芯。

550 这个阈值来自论文中“长寿命 / 短寿命”的分类标准，可以在 `00_config.py` 里修改。

## 6.3 只传电芯 ID 就能预测

```python
scored = fe.score_batch(model_uri=f"models:/{MODEL_NAME}@champion", df=cells)   # cells 里只有 cell_id
```

模型是用 `fe.log_model` 保存的，它记得要从哪张特征表取哪些特征，所以 `score_batch` 会**自动查特征再预测**。这是特征表的核心价值：训练和预测用的特征一定一致。

## 6.4 可以重复运行
notebook 会跳过**已经被当前模型版本打过分**的电芯：

```python
already = spark.table(PREDICTION_TABLE).where(F.col("model_version") == str(champion.version))
cells = cells.join(already, "cell_id", "left_anti")   # 去掉已打过分的
```

所以把它放进定时任务、每天运行，也不会重复写入；没有新电芯时会直接结束。如果上线了新的 champion，所有电芯会用新模型重新打一遍分，预测表里新旧版本的结果都保留，可以对比。

## 6.5 配置定时任务（Jobs）
1. 左侧 **Jobs & Pipelines → Create → Job**；
2. **Task 1**：Notebook 选 `03_features`，计算选 Serverless；
3. **Task 2**：Notebook 选 `06_batch_score`，**Depends on** 选 Task 1，参数 `dataset_role = new_batch`；
4. **Schedule**：例如每天 06:00；**Notifications**：失败时发邮件；
5. 点 **Run now** 运行一次，在 Run 详情里查看每个任务的输出。

两个任务的依赖关系就是真实的流水线：先把新电芯的特征算好，再打分。

学完记得**暂停或删除**这个 Job。

## 6.6 检查点
- [ ] `cell_life_predictions` 表中有 new_batch 电芯的预测
- [ ] 能解释三种质检决策的区别
- [ ] 第二次运行 notebook 时提示“没有需要打分的电芯”
- [ ] （可选）Job 成功运行，两个任务按顺序执行

➡️ 下一步：[阶段 7：效果监控](07-monitoring.md)
