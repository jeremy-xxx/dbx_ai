# 阶段 0：环境准备

## 0.1 你需要什么

| 项目 | 要求 | 说明 |
|---|---|---|
| Databricks Workspace | 开启 **Unity Catalog** | 所有模型、特征表、函数都注册在 UC 里。2023 年后新建的 workspace 基本默认开启 |
| 计算资源 | **Serverless notebook**（推荐）或 **ML Runtime 15.4 LTS+** 的集群 | Serverless 最省事；用经典集群时选 “Machine Learning” 运行时 |
| 权限 | 对某个 catalog 有 `USE CATALOG` + `CREATE SCHEMA` | 没有就找管理员，或使用系统给你的个人 catalog |
| 基础大模型 | **Catalog → system → ai** 下能看到模型（如 `system.ai.claude-sonnet-4-5`） | 由 Unity Gateway 提供（按 token 计费），阶段 1/7/8 要用 |
| 可选 | OpenAI / Anthropic 等 API Key | 仅阶段 5 “外部模型”选做实验需要 |

### 关于 Databricks Free Edition
Free Edition（免费版）只有 Serverless 计算，配额有限。大部分阶段可以做，但：
- Serving 端点数量/规格受限，建议同一时间只保留 1–2 个端点；
- Agent Bricks、部分 AI Gateway 功能、Provisioned Throughput 可能不可用；
- 遇到“功能不可见”先看 [appendix-faq.md](appendix-faq.md)。

## 0.2 概念速览：Unity Catalog 三级命名

Databricks 里所有 AI 资产都用 `catalog.schema.object` 三级命名：

```
main                              ← catalog（目录）
└── ev_battery_lab                ← schema（库），本项目所有东西都放这里
    ├── vehicles                  ← 表（Delta Table）：车辆档案 + BMS 遥测
    ├── maintenance_actions       ← 表：处置动作规则
    ├── battery_features          ← 特征表（本质也是带主键的 Delta 表）
    ├── battery_risk_model        ← 注册模型（Registered Model）
    ├── get_vehicle_profile       ← 函数（UC Function，Agent 的工具）
    └── policy_docs               ← Volume（存放维保/质保文档）
```

**好处**：权限（GRANT）、血缘（Lineage）、审计全部统一。你会在 Models、Features、Agents 页面反复看到这种命名。

## 0.3 把项目导入 Databricks

### 方式 A：Git folder（推荐）
1. 把本文件夹推到你自己的 GitHub/GitLab 仓库。
2. Databricks 左侧 **Workspace** → 你的用户目录 → 右上 **Create → Git folder** → 填仓库 URL。
3. 之后本地改完 push，Databricks 里点 **Pull** 即可同步。

### 方式 B：直接上传
1. **Workspace** → 你的用户目录 → 右键 **Import** → 选择 **File**，逐个上传 `notebooks/*.py`（会自动识别为 notebook，因为首行是 `# Databricks notebook source`）。
2. `agent.py` 需要作为 **普通文件** 放在同一目录（关键是 `07b` 能 `from agent import AGENT`）。
3. `data/policies/` 下的 md 文件也上传到 `notebooks` 的**同级** `data/policies/` 目录（`07a` 会从 `../data/policies` 读取）。

> 💡 保持目录结构不变：`notebooks/` 与 `data/` 是兄弟目录。

## 0.4 修改全局配置

打开 `notebooks/00_config.py`，至少改这几项：

```python
CATALOG = "main"              # 改成你有权限的 catalog
SCHEMA  = "ev_battery_lab"    # 可以不改
LLM_MODE = "legacy"                                             # 默认：旧版端点（Free Edition 实测可用）
LLM_ENDPOINT_LEGACY = "databricks-meta-llama-3-3-70b-instruct"  # 支持工具调用、返回纯文本
LLM_MODEL_GATEWAY = "system.ai.claude-sonnet-4-5"               # 仅 LLM_MODE = "gateway" 时使用
```

**怎么判断用哪种模式**：打开 **Catalog → system → ai → Services** 标签页。
- 只看到 `bge-large-en` 和一堆 MCP service（Free Edition 常见）→ 用 `legacy`；
- 看到 `claude-*`、`gpt-*` 等聊天类 Model service → 可以用 `gateway`，名字填 `system.ai.<服务名>`。

不确定就用 `docs/appendix-faq.md` 里的 “模型可用性测试” 代码试一下。

### ⚠️ 2026-09 平台变化：基础模型迁移到 Unity Gateway
Serving 页面顶部如果有横幅 “Foundation model serving has moved to Unity Gateway”，说明你的 workspace 已切换到新方式：
- **Serving 页面里不再列出 `databricks-*` 基础模型端点**，这很正常。Serving 页面现在只放你自己部署的模型（阶段 4 的风险模型、阶段 7 的 Agent）。
- 基础模型注册在 Unity Catalog 的 `system.ai` 下，通过 `https://<workspace>/ai-gateway/mlflow/v1` 这个 OpenAI 兼容地址调用。
- 原来的 **AI Gateway** 更名为 **Unity Gateway**，成为统一管理模型、Agent、MCP 的控制台。

本项目的代码已兼容两种方式：`LLM_MODE = "legacy"`（默认，通过横幅里的 legacy experience 仍可使用旧端点）或 `"gateway"`。

如果是 **多人共享的 workspace**，端点名称是全 workspace 唯一的，`00_config.py` 已经自动给端点名加了你的用户名前缀，避免冲突。

## 0.5 生成数据

运行 `notebooks/00_setup_data.py`，它会创建：

| 表 | 行数 | 说明 |
|---|---|---|
| `vehicles` | 5000 | 车辆档案 + BMS 遥测 + 标签 `battery_fault_90d`（字段含义见 README 数据字典） |
| `maintenance_actions` | 9 | 处置动作规则（车型 × 风险等级 → 优先级、动作、工时） |

**先花 5 分钟理解数据**（这是做任何 ML 项目的第一步）。在 notebook 或 SQL Editor 里跑：

```sql
-- 各车型故障率与平均 SOH：物流车为什么更高？
SELECT vehicle_model, battery_chemistry, count(*) n,
       round(avg(battery_fault_90d), 3) fault_rate, round(avg(soh_pct), 1) avg_soh
FROM vehicles GROUP BY ALL ORDER BY fault_rate DESC;

-- 快充占比分桶 vs 故障率
SELECT round(fast_charge_ratio, 1) AS fast_bucket, count(*) n, round(avg(battery_fault_90d), 3) fault_rate
FROM vehicles GROUP BY ALL ORDER BY fast_bucket;

-- 触发安全红线的车
SELECT * FROM vehicles WHERE max_cell_temp_c_30d >= 60 OR cell_voltage_imbalance_mv >= 150;
```

## 0.6 自检

- [ ] **Catalog** 里能看到 `你的catalog.ev_battery_lab.vehicles`
- [ ] 打开表 → **Sample Data** 能看到数据，**Columns** 里能看到字段注释
- [ ] 上面 3 条 SQL 能跑通，并对“物流车风险更高、快充越多风险越高”有直观认识
- [ ] **Catalog → system → ai** 下能找到 `LLM_MODEL_GATEWAY` 配置的模型（legacy 模式则是 Serving 的 legacy 页面里有 `databricks-*` 端点）

全部打勾，进入 [阶段 1：Playground](01-playground.md)。
