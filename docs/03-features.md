# 阶段 3：特征工程

> **目标**：从每个电芯的前 100 个循环提取 19 个特征，写入特征表，让训练和预测用同一套特征。
> **notebook**：`03_features.py` **耗时**：约 30 分钟

---

## 3.1 为什么用特征表

| 不用特征表 | 用特征表 |
|---|---|
| 训练时在 notebook 里算一遍，预测时再算一遍，两处代码可能悄悄不一致 | 训练和预测都从同一张表取特征，**口径一定一致** |
| 不知道哪个模型用了哪些特征 | Features 页面的 **Lineage** 能看到特征来自哪些表、被哪个模型使用 |
| 新电芯来了要重新跑整套流程 | 重跑本 notebook，新电芯的特征**按主键合并**进表里 |

在 Databricks 上，特征表就是**带主键的 Delta 表**，这里主键是 `cell_id`。

## 3.2 特征一览

| 类别 | 特征 | 物理含义 |
|---|---|---|
| **ΔQ(V) 统计量**（最重要） | `dq_log_var`、`dq_log_abs_min`、`dq_log_abs_mean`、`dq_skew`、`dq_kurt` | 放电曲线形状的变化程度（阶段 2 的核心发现） |
| 容量衰减 | `qd_cycle2`、`qd_cycle100`、`qd_max_minus_cycle2`、`fade_slope_2_100`、`fade_intercept_2_100`、`fade_slope_91_100` | 早期容量的水平和下降速度 |
| 充电 / 温度 / 内阻 | `chargetime_avg_first5`、`tmax_max_2_100`、`tmin_min_2_100`、`tavg_sum_2_100`、`ir_cycle2`、`ir_min_2_100`、`ir_diff_100_2` | 发热和内阻增长 |
| 快充策略 | `charge_time_0_80_min` | 测试开始前就已知的设计参数 |

## 3.3 两条关键约束

**1. 只用前 100 个循环**
代码里读循环数据时就加了 `cycle BETWEEN 1 AND 100`。这既是业务要求（只测一周就要出结果），也防止**数据泄漏**：如果用到了第 300 个循环的数据，模型在测试时会显得很准，但实际使用时根本拿不到这些数据。

**判断一个特征能不能用，就问一句：在“测完 100 个循环”这个时间点，这个信息已知吗？**
- `charge_time_0_80_min`：快充策略在测试开始前就定好了 → ✅ 可以用；
- `cycle_life`：要测到寿命结束才知道 → ❌ 绝对不能用，它是要预测的答案。

**2. 缺失值留空，不在这里填补**
个别电芯某些循环的测量值异常（比如内阻记录为 0），计算出的特征就会缺失。我们**不在特征表里填补**，而是交给阶段 4 的模型流水线，用**训练集的中位数**填补。如果在这里用全部数据的中位数填，测试集和新批次的信息就“泄漏”进了训练。

## 3.4 可重复运行的写法

```python
if spark.catalog.tableExists(FEATURE_TABLE):
    fe.write_table(name=FEATURE_TABLE, df=features_sdf, mode="merge")   # 已有：按主键合并
else:
    fe.create_table(name=FEATURE_TABLE, primary_keys=["cell_id"], df=features_sdf, ...)   # 首次：创建
```

这样本 notebook 可以直接作为阶段 6 定时任务的第一步：每次运行都把最新电芯的特征补进表里。

最后还开启了表的 **Change Data Feed**（变更数据流）。只有可选的阶段 9 需要它：把特征表同步到在线特征存储时会用到。

## 3.5 检查点
- [ ] 左侧 **Features** 页面能搜到 `cell_early_features`，有 19 个特征列
- [ ] “含缺失值的特征”打印结果可以理解（少量缺失是正常的）
- [ ] 能说出为什么 `charge_time_0_80_min` 可以作为特征，而 `cycle_life` 不行

➡️ 下一步：[阶段 4：建模实验](04-train.md)
