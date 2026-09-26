# 阶段 9（扩展）：真实数据 —— 用前 100 个循环预测电芯寿命

> **学习目标**
> 1. 把一个真实的公开科研数据集接入 Databricks：**下载 → Volume → 解析 HDF5 → 清洗 → Delta 表**；
> 2. 在真实数据上做 **领域特征工程**（ΔQ(V) 曲线统计量），体会“小样本 + 强特征”的建模方式；
> 3. 复用阶段 2/3/6 的平台流程：特征表 → 实验对比 → 注册模型 → 批量预测；
> 4. 学会诚实地评估：训练/测试/**跨批次测试** 三层划分，以及模型的适用边界；
> 5. （可选）把真实模型的结果作为工具接入阶段 7 的 Agent。
>
> **耗时**：3–4 小时（含下载 ~7.7 GB） **前置**：阶段 0–7 **费用**：notebook 计算 + Volume 存储（约 8 GB）

---

## 9.1 为什么要做这一阶段？和前面有什么不同？

| | 阶段 0–8（合成车队数据） | 阶段 9（真实电芯数据） |
|---|---|---|
| 数据对象 | 5000 辆车，车辆级 BMS 汇总指标 | 124 个实验室电芯，**每个循环的完整电压/容量曲线** |
| 任务 | 分类：90 天内是否故障 | 回归：总循环寿命（150–2300 次） |
| 难点 | 平台功能串联 | 数据接入、信号特征提取、小样本建模 |
| 业务场景 | 售后运维 | **电芯来料检验 / 快充策略研发**：只测 100 个循环（几天）就预测寿命，不必等几个月 |

两者 **并存而不是替换**：合成数据证明你能搭完整的业务系统，真实数据证明你能处理真实的电池数据。放在作品集里是互补的。

## 9.2 数据集介绍

**Severson, K.A., Attia, P.M., et al. *Data-driven prediction of battery cycle life before capacity degradation.* Nature Energy 4, 383–391 (2019).**
MIT、Stanford 与丰田研究院（TRI）合作发布。

| 项目 | 内容 |
|---|---|
| 电芯 | A123 APR18650M1A，LFP 正极 / 石墨负极，额定 1.1 Ah、3.3 V |
| 实验条件 | 30℃ 恒温箱；一步或两步快充策略 `C1(Q1)-C2`（到 80% SOC 后 1C 恒流恒压）；4C 放电；电压 2.0–3.6 V |
| 寿命定义 | 容量衰减到额定值 80%（0.88 Ah）时的循环次数 |
| 规模 | 3 个批次（按实验开始日期），共 124 个可用电芯 |
| 格式 | 每批一个 MATLAB v7.3 `.mat`（本质是 HDF5）：2.82 GB / 1.87 GB / 3.01 GB |
| 许可 | **CC BY 4.0**：可自由使用，需署名并引用论文 |
| 主页 | https://data.matr.io/1/projects/5c48dd2bc625d700019f3204 |
| 官方加载代码 | https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation |

**论文结论**（摘要）：只用前 100 个循环的数据，最佳模型的寿命预测测试误差为 **9.1%**；只用前 5 个循环做“长寿/短寿”二分类，误差为 4.9%。
> 注意：论文建模代码需要向作者申请学术许可，本项目的特征和模型是 **我们自己根据论文思路实现的**，结果不会与论文完全一致，这很正常。

**数据的已知问题**（来自各批次页面，notebook 已处理）：
- batch1：部分电芯未循环到 80% 就停止；5 个电芯的测试延续到了 batch2；
- batch2：channel 10 的电芯很快失效，可能有缺陷；
- batch3：部分通道噪声大或提前终止；
- 温度测量不完全可靠（热电偶接触会变化）；cycle 1 数据不在结构体中。

## 9.3 数据接入（`09a_real_data_ingest.py`）

```
data.matr.io ──HTTP 流式下载──► UC Volume: raw_battery_data/*.mat (~7.7 GB)
                                     │  h5py 按需读取（不整体载入内存）
                                     ▼
            cell_metadata        每电芯一行：批次、充电策略、寿命、train/test/secondary_test
            cell_cycle_summary   每电芯每循环一行：放电/充电容量、内阻、温度、充电时间（~10 万行）
            cell_early_curves    第 10 / 100 循环的 Qdlin 曲线（各 1000 点，array<double>）
```

### 下载方式
- **默认**：notebook 直接从 data.matr.io 流式下载到 Volume（10–40 分钟）。
- **workspace 不能访问外网**：本地浏览器下载后，用 Databricks CLI 上传：
  ```bash
  databricks fs cp ./2017-05-12_batchdata_updated_struct_errorcorrect.mat dbfs:/Volumes/<catalog>/ev_battery_lab/raw_battery_data/
  ```
- **快速试跑**：widget `batches` 只填 `2017-06-30`，只下载 1.87 GB（跨批次的 5 个电芯会自动剔除）。

### 解析要点：MATLAB v7.3 结构体
```python
with h5py.File(path, "r") as f:
    batch = f["batch"]                                        # 结构体数组
    cycle_life = f[batch["cycle_life"][i, 0]][()]             # 字段存的是“对象引用”，要 f[ref] 再取值
    policy = f[batch["policy_readable"][i, 0]][()].tobytes()[::2].decode()   # MATLAB 字符串是 UTF-16
    summ = f[batch["summary"][i, 0]]                          # 每循环汇总
    qd = np.hstack(summ["QDischarge"][0, :].tolist())
    cycles = f[batch["cycles"][i, 0]]                         # 每循环的原始曲线
    q100 = np.hstack(f[cycles["Qdlin"][99, 0]][()])           # 只读需要的那两条曲线
```
- 字段名参考官方 `BuildPkl_Batch1.ipynb`；新版 h5py 用 `[()]` 取代旧代码中的 `.value`。
- **只读需要的部分**：原始每循环的电流/电压/时间序列很大，我们只读 summary 和两条插值曲线，几分钟就能解析完。

### 清洗规则（与官方 `Load Data.ipynb` 一致）
| 规则 | 电芯 |
|---|---|
| 未到 80% 容量即停止，剔除 | b1c8, b1c10, b1c12, b1c13, b1c22 |
| batch1 → batch2 续测：寿命 + 续测循环数，并删除 batch2 中的重复记录 | b1c0–4 ↔ b2c7/8/9/15/16（+662/981/1060/208/482） |
| 噪声大或提前终止，剔除 | b3c2, b3c23, b3c32, b3c37, b3c42, b3c43 |

> 💡 巧妙之处：我们只用前 100 个循环做特征，续测电芯的前 100 个循环都在 batch1 里，所以 **只需修正标签**，不用拼接两个文件的循环数据。

### 数据集划分（沿用论文，便于对比）
| 划分 | 来源 | 作用 |
|---|---|---|
| train | batch1+2 奇数位 | 训练 |
| test | batch1+2 偶数位（及第 84 个） | 主测试集 |
| secondary_test | batch3 全部 | **跨批次测试**：不同时间开始的实验，检验泛化 |

## 9.4 特征工程：ΔQ(V)（`09b` 第 1–2 节）

前 100 个循环里容量几乎没降，直接看容量是预测不了寿命的。论文的关键洞察是看 **放电曲线形状的变化**：

```
Qdlin：把放电容量按电压 3.6V→2.0V 均匀插值成 1000 点
ΔQ(V) = Q₁₀₀(V) − Q₁₀(V)       ← 同一电压下，第100次比第10次少放了多少电
特征  = log10( Var(ΔQ(V)) )    ← 这条差值曲线“起伏越大”，寿命越短
```
物理直觉：衰减机制（如锂损失、活性材料损失）会在特定电压区间改变放电曲线形状，早于总容量下降被观察到。

本项目提取的特征（**全部只用前 100 个循环**，防止信息泄漏）：

| 类别 | 特征 |
|---|---|
| ΔQ(V) 统计量 | `dq_log_var`、`dq_log_abs_min`、`dq_log_abs_mean`、`dq_skew`、`dq_kurt` |
| 容量衰减 | `qd_cycle2`、`qd_cycle100`、`qd_max_minus_cycle2`、`fade_slope_2_100`、`fade_intercept_2_100`、`fade_slope_91_100` |
| 充电 / 温度 / 内阻 | `chargetime_avg_first5`、`tmax_max_2_100`、`tmin_min_2_100`、`tavg_sum_2_100`、`ir_cycle2`、`ir_min_2_100`、`ir_diff_100_2` |

特征写入特征表 `cell_early_features`（PK `cell_id`），再用 `FeatureLookup` 组装训练集 —— 复习阶段 6。

## 9.5 建模与实验（`09b` 第 4–6 节）

- **经典一图**：`log Var(ΔQ)` vs `寿命`（对数轴），你应该能看到明显的负相关。
- **实验矩阵**：3 组特征（variance / discharge / full）× 2 种模型（ElasticNet / GBDT）= 6 个 Run。
- **目标变换**：预测 `log10(cycle_life)`，评估时换回循环数 —— 寿命跨度 150–2300，取对数让误差更均衡。
- **指标**：`*_rmse_cycles`（循环数误差）、`*_mape_pct`（平均百分比误差），分别在 train / test / secondary_test 上计算。

**你大概会观察到**（具体数值取决于实现细节）：
1. 单一特征 `dq_log_var` 就已经相当有效 —— 好特征胜过复杂模型；
2. GBDT 训练误差很低、测试误差不低 —— **124 个样本上的过拟合**；正则化线性模型更稳；
3. `secondary_test` 误差通常高于 `test` —— **批次间差异（分布漂移）** 是真实数据的常态，这也是为什么要单独留一个跨批次测试集。

最佳模型（按 test MAPE）注册为 `cell_cycle_life_model@champion`，模型描述中写明了 **适用范围**。

## 9.6 批量预测与 UC 函数（`09b` 第 7–8 节）

- 结果表 `cell_life_predictions`：每个电芯的预测寿命、实际寿命、误差、模型版本；
- UC 函数 `get_cell_life_prediction(cid)`：返回 JSON，可在 SQL、Playground、Agent 中复用。

## 9.7 （可选）接入阶段 7 的 Agent

让售后 Agent 也能回答“实验室电芯 b1c5 的寿命预测是多少？”。改动三处：

**① `07b` 的 `agent_config` 增加**：
```python
"fn_cell_life": FN_CELL_LIFE,
```

**② `agent.py` 的 `TOOLS` 增加一项，`call_tool` 增加一个分支**：
```python
FN_CELL_LIFE = config.get("fn_cell_life")

TOOLS.append({
    "type": "function",
    "function": {
        "name": "get_cell_life_prediction",
        "description": "查询实验室电芯（公开数据集 A123 LFP 18650）基于前100循环的寿命预测。只适用于电芯ID如 b1c5，不适用于车辆ID。",
        "parameters": {"type": "object",
                       "properties": {"cell_id": {"type": "string", "description": "电芯ID，例如 b1c5"}},
                       "required": ["cell_id"]},
    },
})

# call_tool 中：
if name == "get_cell_life_prediction":
    return self._uc(FN_CELL_LIFE, {"cid": args["cell_id"]}) or "未找到该电芯"
```

**③ `07b` 的 `resources` 增加**：`DatabricksFunction(function_name=FN_CELL_LIFE)` 与 `DatabricksTable(table_name=CELL_PRED_TABLE)`。

然后重跑 `07b`：在评估集中加一条 `“电芯 b1c5 预计能循环多少次？”`，以及一条 **越界问题** `“车辆 V00042 的电池还能用多少个循环？”` —— 用 `Guidelines` 检查 Agent 不会把实验室电芯模型错误地套用到整车电池包上。

## 9.8 练习

1. **只用前 N 个循环**：把 `LATE_IDX` 改成 49（用第 10 与第 50 循环），重新提取特征建模。N 越小误差如何变化？这就是“测试时间 vs 预测精度”的工程权衡。
2. **加入充电策略**：从 `charge_policy` 解析出 C1、Q1、C2 三个数值作为特征。精度提升了吗？思考：它在实际来料检验场景中是否可用？
3. **分类任务**：以寿命 550 次为界，把问题改成“长寿/短寿”分类，用 `LogisticRegression` + 前 5 个循环的特征，对比论文的 4.9%。
4. **把模型部署成端点**：复用阶段 4 的代码部署 `cell_cycle_life_model@champion`，并用阶段 5 的方法开启推理表。
5. **换数据集**：用 NASA PCoE 电池老化数据（不同电芯、不同工况）做 SOH 估计，体会不同数据集之间的迁移难度。

## 9.9 检查点

- [ ] Volume 中有 3 个 `.mat` 文件，`cell_metadata` 中有约 124 个电芯
- [ ] 画出了容量衰减曲线和 `log Var(ΔQ)`–寿命散点图
- [ ] Experiments 中有 6 个 Run，能解释 test 与 secondary_test 误差的差异
- [ ] `cell_cycle_life_model@champion` 已注册，描述中写明适用范围
- [ ] （可选）Agent 能正确区分“电芯 ID”与“车辆 ID”

## 9.10 常见坑

| 现象 | 原因 / 解决 |
|---|---|
| 下载很慢或中断 | 重跑单元会跳过已完成的文件；`.part` 文件是未完成的，可删除后重试；或本地下载后用 CLI 上传 |
| `OSError: Unable to open file` | 文件未下载完整；检查文件大小是否与 9.2 表中一致 |
| h5py 读取 Volume 很慢 | Volume 通过 FUSE 挂载，随机读有开销；可先 `shutil.copy` 到本地临时目录再解析（注意本地磁盘空间） |
| `AttributeError: 'Dataset' object has no attribute 'value'` | 你参考了旧版官方代码；新版 h5py 用 `ds[()]` |
| 某些电芯 `cycle_life` 为 NaN | 实验未到寿命终点，notebook 已自动剔除 |
| 电芯数与 124 不一致 | 只处理了部分批次，或清洗规则被修改；检查 `cell_metadata` 按 batch 的计数（官方口径为 41 / 43 / 40；若个别电芯在结构体中寿命缺失被兜底规则剔除，会少 1–2 个，不影响学习） |
| 图表中文显示为方框 | matplotlib 默认无中文字体，图内文字用英文 |

## 9.11 引用

使用本数据集的任何产出（博客、作品集、论文）请注明：

> Severson, K.A., Attia, P.M., Jin, N. et al. Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy* 4, 383–391 (2019). Data: https://data.matr.io/1/ (CC BY 4.0)
