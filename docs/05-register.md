# 阶段 5：上线闸门

> **目标**：用规范的方法选模型，通过闸门检查后注册到 Unity Catalog，设为 `@champion`。
> **notebook**：`05_register.py` **耗时**：约 20 分钟

---

## 5.1 选模型：看交叉验证，不看测试集

这是很多人会犯的错：直接挑“测试集误差最小”的模型。

问题在于：8 个模型里挑测试集表现最好的那个，**测试集就被用来做选择了**。它报告的误差会偏乐观：可能只是这个模型碰巧在这批测试电芯上表现好。

正确的做法：

```
只用训练集信息（交叉验证误差）选出候选  ──►  测试集只用来做最后的检查
```

就像考试：平时用练习题（交叉验证）选复习方法，最后只考一次正式考试（测试集）。拿正式考试的分数来挑方法，分数就失去了公正性。

## 5.2 闸门检查

| 检查项 | 规则 | 防止什么 |
|---|---|---|
| 精度达标 | `test_mape ≤ 15%` | 精度不满足业务要求 |
| 交叉验证稳定 | `cv_mape_std ≤ 0.5 × cv_mape` | 结果靠运气 |
| 测试集与交叉验证一致 | `test_mape ≤ 2 × cv_mape` | 训练时的估计和测试表现差太多，说明哪里有问题 |
| 比现有 champion 好 | 已有 champion 时，新模型 `test_mape` 必须更低 | 新模型反而更差 |

任何一项不通过就**不注册**，并打印原因。阈值由业务和团队约定，可以在 `00_config.py` 或 notebook 中调整。

学习时如果闸门没过、又想继续后面的阶段，可以把 `FORCE_REGISTER` 临时改成 `True`。版本上会记录 `gate_passed = False`。

## 5.3 注册与别名

```python
mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)   # 注册的是 Run 中保存的模型
client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
```

- `runs:/<run_id>/model` 中，最后的 `model` 是阶段 4 `fe.log_model(artifact_path="model")` 时起的名字；
- **别名（Alias）** 是指向某个版本的“可移动指针”。下游（阶段 6、7、9）都通过 `models:/...@champion` 引用模型。以后上线新版本或回滚，只需要移动别名，下游代码不用改。

**版本上的标签**会被下游使用：

| 标签 | 谁用 | 用途 |
|---|---|---|
| `test_mape` | 阶段 6 | 作为预测的误差范围 |
| `test_mape` | 阶段 7 | 和新批次的误差对比，判断是否告警 |
| `features` | 阶段 7 | 只对模型实际用到的特征做漂移检查 |

## 5.4 Run 和模型的关系
- **Run**：一次训练记录，留在 Experiments 里；
- **注册模型的版本**：从 Run 中保存的模型注册而来，进入 Unity Catalog；
- 模型版本页面的 **Source run** 链接可以跳回训练记录，这就是模型的血缘。

## 5.5 检查点
- [ ] 左侧 **Models** 中出现 `cell_cycle_life_model`，最新版本带 `@champion`
- [ ] 版本的 Tags 里有 `test_mape`、`features`、`gate_passed`
- [ ] 能解释为什么按 `cv_mape` 选候选，而不是按 `test_mape`

➡️ 下一步：[阶段 6：批量预测](06-batch-score.md)
