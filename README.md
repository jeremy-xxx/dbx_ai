# 电芯寿命早期预测：基于真实数据的 Databricks 实战项目

**只测 100 个循环（约一周），预测一个锂电池电芯最终能用多少个循环（原本要测几个月）。**

本项目用 MIT / Stanford / 丰田研究院公开的真实电池数据，在 Databricks 上搭一条完整、可落地的流水线：从原始实验数据接入，到特征工程、模型实验、上线闸门、按批次自动预测、效果监控，再到让工程师用自然语言查询数据。

---

## 要解决的问题

电池研发和来料检验中，判断一个电芯“能用多久”通常要把它反复充放电，直到容量衰减到 80%。这要几百到两千多个循环，**耗时几个月**。

论文 Severson et al.（*Nature Energy*, 2019）发现：**只看前 100 个循环的放电曲线变化，就能较准确地预测最终寿命**，测试误差约 9%。这能把寿命评估从几个月缩短到一周，用于：
- **来料检验**：新到一批电芯，快速识别出短寿命的不良品；
- **快充策略研发**：快速评估新的快充方案会让电池寿命缩短多少。

本项目把这个研究结论做成一个可运行、可监控的系统。

## 数据集

| 项目 | 内容 |
|---|---|
| 来源 | Severson, K.A., Attia, P.M., et al. *Data-driven prediction of battery cycle life before capacity degradation.* Nature Energy 4, 383–391 (2019) |
| 电芯 | 124 个 A123 APR18650M1A，LFP 正极 / 石墨负极，额定 1.1 Ah |
| 实验 | 30℃ 恒温箱，几十种一步或两步快充策略（通常每种 2 个电芯），4C 放电，循环到容量 80% |
| 寿命范围 | 约 150–2300 个循环 |
| 规模 | 3 个批次，共约 7.7 GB 的 MATLAB 文件 |
| 下载 | https://data.matr.io/1/ |
| 许可 | **CC BY 4.0**：可自由使用，需署名并引用论文 |

## 流水线

```
data.matr.io ──① 接入──► Volume + Delta 表 ──② 探索分析
                              │
                         ③ 特征工程 ──► 特征表（Features）
                              │
                         ④ 建模实验（Experiments）──► ⑤ 上线闸门（Models：@champion）
                                                              │
   新一批电芯测完 100 个循环 ──► ⑥ 批量预测 + 质检决策（Jobs 定时运行）──► 预测表
                                                              │
                         真实寿命出来后 ──► ⑦ 效果监控：误差、漏检、特征漂移
                                                              │
                              工程师提问 ──► ⑧ Genie 数据助手（自然语言 → SQL）

（可选）⑨ 实时 API：在线特征存储 + Serving 端点
```

**数据集角色**（模拟真实的研发流程）：

| 角色 | 来源 | 在流水线中的用途 |
|---|---|---|
| `train` | batch1 + batch2 的一半 | 训练模型 |
| `test` | batch1 + batch2 的另一半 | 评估模型，决定能否上线 |
| `new_batch` | batch3（晚近一年开始的实验） | **模拟上线后新到的一批电芯**：阶段 6 打分，阶段 7 检验模型的真实表现 |

## 用到的 Databricks 功能

只用这个场景**真正需要**的功能：

| 功能 | 在项目中的作用 | 阶段 |
|---|---|---|
| Unity Catalog、Volume、Delta 表 | 存放原始文件、清洗后的数据、预测结果，统一权限和血缘 | 1 |
| **Features**（特征表） | 训练和预测共用同一套特征；新电芯的特征按主键合并 | 3 |
| **Experiments**（MLflow） | 对比 8 组实验，记录交叉验证和测试误差 | 4 |
| **Models**（UC 模型注册） | 闸门检查后设为 `@champion`，下游只认别名 | 5 |
| **Jobs** | 按批次自动运行“算特征 → 打分” | 6 |
| Genie Agents | 工程师用自然语言查询电芯数据 | 8 |
| Serving + 在线特征存储（可选） | 测试台软件实时调用 | 9 |

## 目录结构

```
databricks-aiml-lab/
├── README.md
├── docs/                         教学文档（按阶段阅读）
│   ├── 00-setup.md               环境准备（必读）
│   ├── 01-ingest.md
│   ├── 02-eda.md
│   ├── 03-features.md
│   ├── 04-train.md
│   ├── 05-register.md
│   ├── 06-batch-score.md
│   ├── 07-monitoring.md
│   ├── 08-genie.md
│   ├── 09-serving.md             可选
│   └── appendix-faq.md           常见问题
└── notebooks/                    在 Databricks 中运行（Databricks 源码格式 .py）
    ├── 00_config.py              全局配置，其它 notebook 用 %run 引用
    ├── 01_ingest.py
    ├── 02_eda.py
    ├── 03_features.py
    ├── 04_train.py
    ├── 05_register.py
    ├── 06_batch_score.py
    ├── 07_monitoring.py
    ├── 08_genie.py
    ├── 09_serving.py             可选
    └── 99_cleanup.py             清理资源
```

## 学习路线

| 阶段 | 内容 | 预计耗时 | 会持续计费的资源 |
|---|---|---|---|
| 0 环境准备 | 导入项目、修改配置 | 20 min | — |
| 1 数据接入 | 下载 7.7 GB、解析 HDF5、清洗、写表 | 1 h（含下载） | Volume 存储 |
| 2 探索分析 | 寿命分布、快充影响、ΔQ(V) 的发现 | 45 min | — |
| 3 特征工程 | 提取 19 个特征、写入特征表 | 30 min | — |
| 4 建模实验 | 4 组特征 × 2 种模型，交叉验证 | 45 min | — |
| 5 上线闸门 | 按交叉验证选模型、闸门检查、注册 | 20 min | — |
| 6 批量预测 | 给新批次打分、质检决策、配置 Job | 30 min | Job（按运行计费） |
| 7 效果监控 | 误差、漏检、特征漂移、再训练 | 45 min | — |
| 8 Genie 数据助手 | 宽表视图、业务说明、创建 Genie Agent | 45 min | — |
| 9 实时 API（可选） | 在线特征存储 + Serving 端点 | 1 h | ⚠️ 在线存储、Serving 端点 |

> 💰 学完运行 `99_cleanup.py`；暂停或删除阶段 6 创建的 Job。

## 如何开始

1. 读 [docs/00-setup.md](docs/00-setup.md)，用 Git folder 把本仓库导入 Databricks；
2. 按需修改 `notebooks/00_config.py` 里的 `CATALOG`；
3. 从 `01_ingest.py` 开始，按顺序运行，每个阶段先读对应文档；
4. notebook 中 **`# 👀 界面：`** 开头的注释说明这行代码运行后**去界面哪里看效果**；**`# （界面无变化）`** 表示只改了当前会话的设置。

## 版本说明

- 按 2026 年 11 月的 Databricks 产品与官方文档编写：MLflow 3、Unity Catalog 模型注册、Feature Engineering、Genie Agents（原 Genie Spaces）。
- Databricks 更新很快，按钮名称可能和文档略有不同，以功能含义为准；遇到对不上的情况看 [appendix-faq.md](docs/appendix-faq.md)。
- 使用本数据集的任何产出（博客、作品集、论文）请注明出处：Severson et al., *Nature Energy* 4, 383–391 (2019)，数据来自 https://data.matr.io/1/（CC BY 4.0）。
