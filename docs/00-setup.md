# 阶段 0：环境准备

## 0.1 你需要什么

| 项目 | 要求 | 说明 |
|---|---|---|
| Databricks Workspace | 开启 Unity Catalog | Free Edition 也可以完成阶段 1–8 |
| 计算资源 | Serverless notebook（推荐） | 所有 notebook 开头都会 `%pip install` 所需的包 |
| 权限 | 某个 catalog 的 `USE CATALOG` + `CREATE SCHEMA` | 本项目在其下创建 `battery_life_lab` |
| SQL Warehouse | Pro 或 Serverless 的 SQL Warehouse | 阶段 8 Genie 需要；Free Edition 自带 Serverless Starter Warehouse |
| 外网访问 | 能访问 `data.matr.io` | 阶段 1 下载数据；不能访问时可本地下载后上传 |
| 存储空间 | 约 8 GB | 原始数据，解析完可删除 |

## 0.2 Unity Catalog 中会创建哪些东西

```
main                              ← catalog
└── battery_life_lab              ← schema：本项目的所有对象
    ├── raw_battery_data          ← Volume：原始 .mat 文件（阶段 1）
    ├── cell_metadata             ← 表：电芯元数据、快充策略、寿命、数据集角色（阶段 1）
    ├── cell_cycle_summary        ← 表：每电芯每循环的汇总数据（阶段 1）
    ├── cell_early_curves         ← 表：第 10 / 100 循环的放电曲线（阶段 1）
    ├── cell_early_features       ← 特征表（阶段 3）
    ├── cell_cycle_life_model     ← 注册模型（阶段 5）
    ├── cell_life_predictions     ← 表：批量预测结果（阶段 6）
    ├── model_monitoring          ← 表：监控记录（阶段 7）
    └── cell_overview             ← 视图：给 Genie 用的宽表（阶段 8）
```

## 0.3 导入项目

推荐用 **Git folder**：左侧 **Workspace** → 你的用户目录 → 右上 **Create → Git folder** → 填仓库地址。之后仓库有更新，在 Git folder 里点 **Pull** 即可同步。

如果提示 GitHub 认证失败：需要在 GitHub 上给 Databricks GitHub App 授权访问这个仓库，或在 Databricks 的 **Settings → Linked accounts** 里改用个人访问令牌。

## 0.4 修改配置

打开 `notebooks/00_config.py`，通常只需要改一处：

```python
CATALOG = "main"              # 改成你有权限的 catalog（Free Edition 常见为 "workspace"）
SCHEMA = "battery_life_lab"   # 可以不改
```

其它可调的业务参数也在这个文件里：

| 参数 | 默认 | 含义 |
|---|---|---|
| `EARLY_CYCLES` | 100 | 用前多少个循环做预测 |
| `SHORT_LIFE_THRESHOLD` | 550 | 寿命低于多少算“短寿命”电芯 |
| `MAX_ACCEPTABLE_MAPE` | 15.0 | 模型上线要求：测试集平均百分比误差上限 |
| `DRIFT_ALERT_RATIO` | 1.5 | 新批次误差超过测试集误差多少倍时告警 |

### `00_config.py` 是做什么的
它是全项目的公共配置，**本身不处理数据**。每个 notebook 开头都有一行 `%run ./00_config`，会自动执行它。运行时会定义表名、模型名等变量，并创建 schema（已存在则跳过，不删数据）。

注意：
- 每个 notebook 有自己独立的 Python 会话，所以每个 notebook 都要自己 `%run ./00_config`；
- `%run` 要单独占一个单元格，放在 `%pip install` 和 `dbutils.library.restartPython()` **之后**，否则重启会清空刚加载的变量。

## 0.5 自检
- [ ] 运行 `00_config.py`，打印出 `schema = main.battery_life_lab`
- [ ] Catalog 中出现 `battery_life_lab`

➡️ 下一步：[阶段 1：数据接入](01-ingest.md)
