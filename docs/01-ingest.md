# 阶段 1：数据接入

> **目标**：把 7.7 GB 的科研原始数据可靠地变成三张干净、带说明的 Delta 表。
> **notebook**：`01_ingest.py` **耗时**：约 1 小时（主要是下载）

---

## 1.1 数据集介绍

**Severson, K.A., Attia, P.M., et al. *Data-driven prediction of battery cycle life before capacity degradation.* Nature Energy 4, 383–391 (2019)**，MIT、Stanford 与丰田研究院（TRI）合作发布。

| 项目 | 内容 |
|---|---|
| 电芯 | A123 APR18650M1A，LFP 正极 / 石墨负极，额定 1.1 Ah、3.3 V |
| 实验条件 | 30℃ 恒温箱；一步或两步快充策略 `C1(Q1)-C2`（到 80% 电量后 1C 恒流恒压）；4C 放电；电压 2.0–3.6 V |
| 寿命定义 | 容量衰减到额定值 80%（0.88 Ah）时的循环次数 |
| 规模 | 3 个批次（按实验开始日期），清洗后共 124 个电芯 |
| 格式 | 每批一个 MATLAB v7.3 `.mat` 文件：2.82 GB / 1.87 GB / 3.01 GB |
| 许可 | CC BY 4.0 |
| 主页 | https://data.matr.io/1/projects/5c48dd2bc625d700019f3204 |
| 官方加载代码 | https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation |

**快充策略 `C1(Q1%)-C2` 怎么读**：以 `5.4C(40%)-3.6C` 为例，先用 5.4C 的电流充到 40% 电量，再用 3.6C 充到 80%。“1C”指一小时充满的电流，5.4C 就是 1C 的 5.4 倍，约 11 分钟能充满。

## 1.2 数据流

```
data.matr.io ──HTTP 流式下载──► Volume: raw_battery_data/*.mat
                                     │  h5py 按需读取
                                     ▼
          cell_metadata        每电芯一行：批次、快充策略（解析为 c1/q1/c2）、寿命、数据集角色
          cell_cycle_summary   每电芯每循环一行：放电/充电容量、内阻、温度、充电时间（约 10 万行）
          cell_early_curves    第 10 / 100 循环的放电容量-电压曲线（各 1000 点）
```

## 1.3 下载
- 默认从 data.matr.io 流式下载到 Volume，每次写 16 MB，先写成 `.part` 文件，下载完整后才改名，中断后重跑会跳过已完成的文件；
- **workspace 不能访问外网**：本地浏览器下载后用 Databricks CLI 上传：
  ```bash
  databricks fs cp ./2017-05-12_batchdata_updated_struct_errorcorrect.mat dbfs:/Volumes/main/battery_life_lab/raw_battery_data/
  ```

## 1.4 解析 MATLAB v7.3 文件
`.mat` v7.3 本质是 HDF5 格式，用 `h5py` 读取。结构体里的字段存的是**对象引用**，要用 `f[ref]` 再取一次值：

```python
with h5py.File(path, "r") as f:
    batch = f["batch"]
    cycle_life = f[batch["cycle_life"][i, 0]][()]                              # 引用 → 实际值
    policy = f[batch["policy_readable"][i, 0]][()].tobytes()[::2].decode()     # MATLAB 字符串是 UTF-16
```

原始文件里还有每个循环的电流、电压、时间序列，数据量很大。我们**只读需要的部分**：每循环的汇总数据，加上第 10、100 循环的两条插值曲线。这样几分钟就能解析完。

## 1.5 清洗规则
与官方 `Load Data.ipynb` 一致：

| 规则 | 电芯 |
|---|---|
| 没循环到 80% 容量就停止了，剔除 | b1c8, b1c10, b1c12, b1c13, b1c22 |
| batch1 中途停止、在 batch2 继续测试：寿命加上续测的循环数，删除 batch2 中的重复记录 | b1c0–4 ↔ b2c7/8/9/15/16（+662/981/1060/208/482） |
| 噪声大或提前终止，剔除 | b3c2, b3c23, b3c32, b3c37, b3c42, b3c43 |

我们只用前 100 个循环做特征，续测电芯的前 100 个循环都在 batch1 里，所以**只需要修正寿命标签**，不用拼接两个文件的数据。

## 1.6 数据集角色

| 角色 | 来源 | 用途 |
|---|---|---|
| `train` | batch1+2 按顺序交替取一半（沿用论文的划分方式） | 训练 |
| `test` | batch1+2 的另一半 | 评估、上线闸门 |
| `new_batch` | batch3 全部 | 模拟上线后新到的电芯 |

batch3 比前两批晚开始近一年，这正好模拟真实情况：模型上线后遇到的数据，和训练时的数据并不完全一样。

## 1.7 表和列的注释
notebook 最后给表和关键列写了中文注释，比如 `cycle_life`、`charge_policy`、`dataset_role` 是什么意思。这些注释：
- 在 Catalog 的 **Columns** 标签里能看到，方便别人理解数据；
- **阶段 8 的 Genie 会读取它们**，帮助它把问题翻译成正确的 SQL。

## 1.8 检查点
- [ ] Volume `raw_battery_data` 里有 3 个 `.mat` 文件，大小与 1.1 节一致
- [ ] `cell_metadata` 各批次电芯数为 41 / 43 / 40（个别电芯寿命缺失时会少 1–2 个）
- [ ] `dataset_role` 有 train / test / new_batch 三种值
- [ ] “策略解析失败的电芯”打印为空列表

## 1.9 常见坑

| 现象 | 处理 |
|---|---|
| 下载很慢或中断 | 重跑会跳过已完成的文件；`.part` 是未完成的文件，可删除后重试 |
| `OSError: Unable to open file` | 文件没下载完整，检查大小 |
| `AttributeError: 'Dataset' object has no attribute 'value'` | 参考的是旧版官方代码，新版 h5py 用 `ds[()]` |
| 策略解析失败的电芯不为空 | 个别策略写法不同；它们的 `c1/q1/c2` 为空，不影响后续建模 |

➡️ 下一步：[阶段 2：探索分析](02-eda.md)
