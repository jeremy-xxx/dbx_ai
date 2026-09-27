# Databricks AI/ML 功能区实战：电动车电池健康预测 + 售后运维 Agent

本项目用 **一个贯穿始终的电动车（EV）电池业务故事**，带你把 Databricks 左侧 **AI/ML** 功能区的 7 个入口全部走一遍：

| 侧边栏入口 | 它是干什么的（一句话） | 本项目对应阶段 |
|---|---|---|
| **Playground** | 在网页里直接和各种大模型聊天、对比、挂工具，零代码试 prompt | 阶段 1 |
| **Experiments** | MLflow 实验追踪：记录每次训练的参数/指标/模型，以及 LLM/Agent 的 Trace | 阶段 2 |
| **Models** | Unity Catalog 模型注册表：模型版本、别名（champion/challenger）、权限、血缘 | 阶段 3 |
| **Serving** | 把模型/LLM 部署成 REST 接口（Model Serving Endpoint），也包含官方托管的基础大模型 | 阶段 4 |
| **AI Gateway**（2026-09 起更名 **Unity Gateway**） | 模型/端点的统一网关：基础模型入口、限流、用量追踪、请求日志（推理表）、安全护栏、回退 | 阶段 5 |
| **Features** | 特征工程 / 特征表：把特征集中管理、复用，训练与推理自动查找特征 | 阶段 6 |
| **Agents** | 构建 AI Agent：Agent Bricks（低代码）+ Agent Framework（代码）+ 评估 + 部署 | 阶段 7 |

最后在 **阶段 8（综合项目）** 把它们串起来做一条“车队电池风险巡检 → 自动生成进站通知与技师工单”的流水线。

**阶段 9（扩展）** 接入真实公开数据集（MIT/Stanford/TRI 124 个电芯的循环寿命数据，Nature Energy 2019），用前 100 个循环预测电芯寿命，复用同一套平台流程。

---

## 业务故事

> 你是一家电动车公司 **电池售后与车队运维部门** 的数据/AI 工程师。每辆车的 BMS（电池管理系统）会定期回传遥测数据：健康度 SOH、电芯压差、最高温度、快充占比、循环次数、告警次数……公司希望：
> 1. **预测哪些车的电池在未来 90 天内可能发生故障**（传统机器学习：分类模型）；
> 2. 给售后服务顾问配一个 **AI 助手**：输入车辆 ID，它能查车辆档案、调用风险模型打分、判断是否在质保期内、再按《电池维保规程》给出处置建议和对车主的沟通话术（生成式 AI / Agent）；
> 3. 所有模型和大模型调用都要 **可追踪、可治理、可控成本**，并且 **安全红线（过温、压差过大）绝不能漏报**。

数据全部由脚本 **合成生成**（5000 辆车），规律符合常识：循环多、快充多、高温地区、NMC 三元电池、压差大 → 衰减更快、故障风险更高。不需要任何外部数据源。

```
                    ┌──────────── 阶段1 Playground：先用 UI 试 prompt（车主反馈分类）──────┐
                    │                                                                     │
 合成 BMS 遥测数据 ──► 阶段2 Experiments 训练多个模型 ──► 阶段3 Models 注册 champion         │
 (Delta 表)                                                 │                             │
     │                                                      ▼                             │
     │                                       阶段4 Serving：电池风险模型 REST 端点          │
     │                                                      │                             │
     │                                       阶段5 AI Gateway：限流/日志/用量               │
     ▼                                                      │                             ▼
 阶段6 Features：电池特征表 + FeatureLookup ─────────► 阶段7 Agents：电池售后运维 Agent
                                                   (UC 函数工具 + 调用风险端点 + 大模型 + 维保规程)
                                                            │
                                         阶段8 综合：车队批量巡检 + ai_query 生成进站通知 + 工单
```

---

## 目录结构

```
databricks-aiml-lab/
├── README.md                        ← 你正在看的总览
├── docs/                            ← 教学文档（按阶段阅读）
│   ├── 00-environment-setup.md      环境准备（必读）
│   ├── 01-playground.md
│   ├── 02-experiments.md
│   ├── 03-models.md
│   ├── 04-serving.md
│   ├── 05-ai-gateway.md
│   ├── 06-features.md
│   ├── 07-agents.md
│   ├── 08-capstone.md
│   ├── 09-real-data.md              扩展：真实电芯循环寿命数据
│   └── appendix-faq.md              常见报错与排查
├── notebooks/                       ← 导入 Databricks 运行的 notebook（Databricks 源码格式 .py）
│   ├── 00_config.py                 全局配置（catalog / schema / 端点名），其它 notebook 用 %run 引用
│   ├── 00_setup_data.py             生成合成 BMS 遥测数据
│   ├── 01_playground_to_code.py
│   ├── 02_experiments_tracking.py
│   ├── 03_models_registry.py
│   ├── 04_model_serving.py
│   ├── 05_ai_gateway.py
│   ├── 06_feature_engineering.py
│   ├── 07a_agent_tools.py           创建 UC 函数工具 + 上传维保/质保文档
│   ├── 07b_agent_build_deploy.py    编写/评估/注册/部署 Agent
│   ├── agent.py                     Agent 源码（MLflow “models from code”）
│   ├── 08_capstone.py
│   ├── 09a_real_data_ingest.py      扩展：下载并解析真实数据集 → Delta 表
│   ├── 09b_cycle_life_model.py      扩展：早期特征 + 寿命预测模型
│   └── 99_cleanup.py                清理资源，避免持续计费
└── data/
    └── policies/                    给 Knowledge Assistant / Agent 用的“公司规程”文档
        ├── battery_maintenance_sop.md   电池维保规程（风险等级、安全红线、沟通要求）
        ├── battery_warranty_policy.md   电池质保政策
        └── charging_faq.md              充电与续航常见问题
```

---

## 数据字典（`vehicles` 表）

| 字段 | 含义 | 说明 |
|---|---|---|
| `vehicle_id` | 车辆 ID | `V00001` ~ `V05000` |
| `owner_name` / `owner_type` | 车主 / 类型 | `private`（私人）或 `fleet`（车队） |
| `vehicle_model` | 车型 | `city_ev` 城市小车 / `suv_ev` SUV / `commercial_van` 物流车 |
| `battery_chemistry` | 电芯化学体系 | `LFP` 磷酸铁锂 / `NMC` 三元锂 |
| `climate_zone` | 使用地气候 | `cold` / `temperate` / `hot` |
| `vehicle_age_months` | 车龄（月） | |
| `odometer_km` | 总里程 | |
| `cycle_count` | 等效满充循环次数 | |
| `fast_charge_ratio` | 快充电量占比 | 0–1 |
| `avg_depth_of_discharge` | 平均放电深度 | 0–1 |
| `max_cell_temp_c_30d` | 近 30 天电芯最高温度（℃） | ≥ 60 为安全红线 |
| `cell_voltage_imbalance_mv` | 电芯最大压差（mV） | ≥ 150 为安全红线 |
| `bms_alerts_90d` | 近 90 天 BMS 告警次数 | |
| `soh_pct` | 电池健康度 SOH（%） | 新车 100，质保更换门槛 70 |
| **`battery_fault_90d`** | **标签：未来 90 天是否发生电池故障** | 约 12% 为 1 |

---

## 如何开始

1. 读 [docs/00-environment-setup.md](docs/00-environment-setup.md)，把整个 `databricks-aiml-lab` 文件夹导入 Databricks（推荐 Git folder）。
2. 修改 `notebooks/00_config.py` 里的 `CATALOG` / `SCHEMA` / `LLM_MODEL`。
3. 运行 `notebooks/00_setup_data.py` 生成数据。
4. 按阶段顺序阅读 `docs/0X-*.md`，每个文档都会告诉你：**先在 UI 里看什么 → 再跑哪个 notebook → 做哪些练习 → 如何自检**。

## 学习路线与预计耗时

| 阶段 | 难度 | 预计耗时 | 前置 | 会产生持续费用的资源 |
|---|---|---|---|---|
| 0 环境准备 | ★ | 30 min | — | 无 |
| 1 Playground | ★ | 45 min | 0 | 无（按 token 计费） |
| 2 Experiments | ★★ | 1 h | 0 | 无 |
| 3 Models | ★★ | 45 min | 2 | 无 |
| 4 Serving | ★★ | 1 h | 3 | **是：Serving 端点**（开启 scale-to-zero） |
| 5 AI Gateway | ★★★ | 1 h | 4 | 推理表存储（很少） |
| 6 Features | ★★★ | 1 h | 2 | 无 |
| 7 Agents | ★★★★ | 2–3 h | 4, 6 | **是：Agent 端点、Agent Bricks** |
| 8 综合项目 | ★★★★ | 1–2 h | 全部 | 同上 |
| 9 真实数据（扩展） | ★★★★ | 3–4 h | 0–7 | Volume 存储约 8 GB |

> 💰 **学完一定要跑 `99_cleanup.py`** 或手动删除 Serving 端点。

## 为什么主线用合成数据？
主线目标是学平台：合成数据规律明确、有车辆/质保/车主等业务字段，能把 7 个功能区完整串起来，出问题也容易判断是“平台用错了”还是“数据有问题”。
公开电池数据集大多是实验室电芯级数据（样本几十到一百多个，没有车辆和业务字段），所以作为 **阶段 9 扩展** 与主线并存，而不是替换：见 [docs/09-real-data.md](docs/09-real-data.md)。
数据集：Severson et al., *Nature Energy* 4, 383–391 (2019)，https://data.matr.io/1/ ，CC BY 4.0。

## 版本说明

- 编写时参考 Databricks 2025–2026 年的产品形态：MLflow 3、Unity Catalog 模型注册、Mosaic AI Agent Framework、Agent Bricks、AI Gateway。
- Databricks UI 更新很快，按钮名称可能与文档略有不同；**以功能含义为准**，遇到对不上的情况看 [appendix-faq.md](docs/appendix-faq.md)。
- **2026-09 起基础模型迁移到 Unity Gateway**（原 AI Gateway 更名）：模型注册在 `system.ai` 下（如 `system.ai.qwen3-next-80b-a3b-instruct`），不再出现在 Serving 页面。本项目统一通过 Unity Gateway 调用大模型，详见 [docs/00](docs/00-environment-setup.md)。可用模型因区域而异，请在 **AI/ML → Unity Gateway → Models** 确认。
