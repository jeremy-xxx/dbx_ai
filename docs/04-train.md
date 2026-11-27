# 阶段 4：建模实验

> **目标**：用 MLflow 系统地比较 8 种方案，找出在小样本真实数据上**又准又稳**的模型。
> **notebook**：`04_train.py` **耗时**：约 45 分钟

---

## 4.1 实验设计

**4 组特征 × 2 种模型 = 8 个 Run**

| 特征组 | 内容 | 想验证什么 |
|---|---|---|
| `variance` | 只有 `dq_log_var` | 单个强特征能做到多好 |
| `discharge` | ΔQ(V) 统计量 + 早期容量 | 只看放电行为 |
| `full` | 放电 + 充电 + 温度 + 内阻 | 更多信息是否更好 |
| `full_policy` | `full` + 快充策略 | 已知的设计参数有没有帮助 |

| 模型 | 特点 |
|---|---|
| `elasticnet` | 带正则化的线性模型：特征多也不容易过拟合，**小样本下通常更稳** |
| `gbdt` | 梯度提升树：能拟合非线性关系，但训练集只有约 40 个电芯，容易过拟合 |

**预测目标**：`log10(循环寿命)`。寿命从 150 到 2300 跨度很大，直接预测时，误差会被长寿命电芯主导；取对数后，各个电芯的相对误差更均衡。评估时再换算回循环次数。

## 4.2 三个误差指标

| 指标 | 怎么算 | 用来判断 |
|---|---|---|
| `cv_mape` / `cv_mape_std` | 训练集内部 **5 折交叉验证**：分成 5 份，轮流用 4 份训练、1 份验证 | **稳定性**。只有约 40 个训练样本，单次划分很容易受运气影响 |
| `train_mape` | 训练集上的误差 | 和 `test_mape` 比较，判断是否过拟合 |
| `test_mape` | 测试集上的平均百分比误差 | 阶段 5 上线闸门的主要依据 |

MAPE（平均百分比误差）= 平均的 |预测 − 真实| / 真实 × 100%。比如真实寿命 1000、预测 900，误差就是 10%。

## 4.3 和特征表的配合

```python
lookups = [FeatureLookup(table_name=FEATURE_TABLE, lookup_key="cell_id", feature_names=feats)]
train_set = fe.create_training_set(df=train_labels, feature_lookups=lookups,
                                   label="log_cycle_life", exclude_columns=["cell_id"])
...
fe.log_model(model=model, artifact_path="model", flavor=mlflow.sklearn, training_set=train_set, ...)
```

- `train_labels` 里只有 `cell_id` 和目标值，特征由 `FeatureLookup` 从特征表自动关联进来；
- `fe.log_model` 保存模型时会记住“需要从哪张特征表取哪些特征”。所以阶段 6 预测时**只传电芯 ID**，特征自动查出；Features 页面也能看到模型和特征表之间的血缘。

## 4.4 模型流水线

```python
Pipeline([
    ("impute", SimpleImputer(strategy="median")),   # 缺失值：用训练集的中位数填补
    ("scale", StandardScaler()),                    # 标准化（线性模型需要）
    ("reg", ElasticNetCV(...)),                     # 内部再用交叉验证选正则化强度
])
```

填补和标准化放在流水线里，**只在训练数据上学习参数**，预测时用同样的参数处理新数据，避免泄漏。

### 一个新版 MLflow 的细节
新版 MLflow 默认用更安全的 **skops** 格式保存 sklearn 模型：只接受白名单里的类型，防止加载模型时执行恶意代码。本项目的流水线里有两个类型不在默认白名单中，所以需要显式信任：

```python
skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"]
#                    └ 来自 SimpleImputer   └ 来自 GBDT 的决策树
```

不加的话会报 `MlflowException: The saved sklearn model references untrusted types`，Run 显示为失败。

## 4.5 怎么读结果

| 现象 | 说明 |
|---|---|
| `train_mape` 远小于 `test_mape` | **过拟合**：模型记住了训练数据，泛化差。GBDT 在约 40 个样本上很常见 |
| `cv_mape_std` 很大 | 不稳定：换几个训练电芯，误差就变化很大 |
| 特征多的组反而更差 | 小样本下，多出来的特征可能只是在拟合噪声 |

论文报告的最佳测试误差约 9%。我们的特征和模型是按论文思路自己实现的（论文的建模代码需要学术许可），结果在同一量级就正常。

## 4.6 界面操作
**Experiments → battery-cycle-life**：
1. 点 **Columns**，把 `cv_mape`、`test_mape`、`train_mape` 加到列表里；
2. 勾选几个 Run → **Compare → Scatter**：横轴 `cv_mape`、纵轴 `test_mape`。两者一致的模型更可信；
3. 打开某个 Run → **Artifacts** → `pred_vs_obs.png`，看预测值和真实值的散点图。

## 4.7 检查点
- [ ] 8 个 Run 都是成功状态（绿色 ✓）
- [ ] 能指出哪个模型过拟合最严重，依据是什么
- [ ] 最佳 `test_mape` 和论文的约 9% 处在同一量级（明显高于 20% 的话，检查阶段 1 的清洗和阶段 3 的特征）

➡️ 下一步：[阶段 5：上线闸门](05-register.md)
