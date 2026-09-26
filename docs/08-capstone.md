# 阶段 8：综合项目 —— 车队电池风险巡检流水线

> **学习目标**：把 7 个功能区串成一条每天自动运行的业务流水线，并完成一次完整的 MLOps / LLMOps 复盘。
>
> **耗时**：1–2 小时 **前置**：阶段 1–7

---

## 8.1 架构总览

```
┌───────────────────────────── 每天定时（Jobs）──────────────────────────────────┐
│                                                                               │
│  battery_features ──► battery_risk_model@champion ──► 风险分层 + 安全红线(P0)   │
│   (Features)            (Models, spark_udf)                  │                 │
│                                                              ▼                 │
│                                         匹配处置动作 + 质保判断                  │
│                                                              │                 │
│                          ai_query(LLM) 批量生成车主/车队进站通知                 │
│                           (Serving: Foundation Model)        │                 │
│                                                              │                 │
│                          Top N → 电池售后 Agent 生成技师工单建议                 │
│                           (Agents)                           ▼                 │
│                                            battery_service_campaign 表         │
└───────────────────────────────────────────────────────────────────────────────┘
       ▲ 所有端点调用经过 AI Gateway（限流 / 推理表 / 用量）
       ▲ 所有训练、评估、Trace 记录在 Experiments
       ▲ 所有原型先在 Playground 验证
```

| 功能区 | 在流水线中的角色 |
|---|---|
| Playground | 进站通知 prompt、Agent 工具组合先在这里调试 |
| Experiments | 风险模型训练记录；Agent 评估与 Trace |
| Models | `battery_risk_model@champion` 决定用哪个版本打分；结果表记录 `model_version` 便于追溯 |
| Features | 打分数据来自特征表（最新遥测），与训练口径一致 |
| Serving | LLM 端点（ai_query）、风险端点（Agent 工具）、Agent 端点 |
| AI Gateway | 限流保护预算、推理表留痕审计 |
| Agents | 为最需要处置的车辆生成技师工单建议 |

## 8.2 运行 `08_capstone.py`

notebook 顶部有 3 个 widget：`top_n`（生成通知的车辆数）、`agent_top_n`（交给 Agent 的数量）、`agent_endpoint`（留空自动查找）。

检查每一步的输出：
1. 各车型 × 风险等级的车辆数、红线车数量：物流车是不是最多？
2. `candidates`：**红线车是否都排在最前、优先级为 P0？** 阶段 6 中被“注入异常”的 V00001 应该在里面。
3. `notice_draft`：
   - 红线车的通知是否清楚提示“停止快充、24 小时内进站”？
   - 车队车辆的通知对象是否是调度负责人？
   - 有没有泄露概率、有没有恐慌性措辞？
4. `agent_work_order`：与通知相比，Agent 多了哪些信息？（它会查规程和质保政策，指出审批要求、是否免费等）

## 8.3 把它变成定时任务

1. 左侧 **Jobs & Pipelines → Create → Job**。
2. Task：类型 Notebook，选择 `08_capstone`，计算选 Serverless。
3. Parameters：`top_n = 50`，`agent_top_n = 3`。
4. Schedule：每天 06:00（在车辆出车之前完成巡检）。
5. Notifications：失败时发邮件给自己。
6. **Run now** 手动跑一次，看 Run 详情里的每个单元格输出。

> 进阶：做成多任务 Job —— `06_feature_engineering`（用最新 BMS 数据刷新特征）→ `08_capstone`（打分 + 生成），体会任务依赖。

## 8.4 可观测性复盘

跑完后，逐个回到侧边栏入口，找到这次运行留下的痕迹：

| 入口 | 你应该能看到 |
|---|---|
| **Experiments** | `ev-battery-genai` 中 Agent 调用的 Trace（若 Agent 端点开启了 trace 记录） |
| **Serving** | 风险端点、LLM 端点、Agent 端点的 Metrics 中请求量上升 |
| **AI Gateway** | 用量统计；`risk_ep_payload` 推理表中出现 Agent 发起的请求 |
| **Models** | `battery_risk_model` 的血缘 |
| **Features** | `battery_features` 的 Consumers 中出现本 notebook / Job |

## 8.5 挑战任务（选做，任选 2 个）

1. **闭环再训练**：把推理表中的线上请求 + 90 天后的真实故障结果组合成新训练集，重新训练、注册为 challenger，通过阶段 3 的“AUC + recall 闸门”后自动晋升，最后更新端点配置。
2. **通知质量评估**：把 `notice_draft` 作为输出，用 `mlflow.genai.evaluate` + `Guidelines`（“红线车必须提示停止快充”“不透露概率”“无恐慌措辞”）批量评估，统计违规率。
3. **成本分析**：用 `system.serving.endpoint_usage` 统计本流水线每次运行消耗的 token 数，估算每辆车的 AI 巡检成本。
4. **真实数据**：完成 [阶段 9](09-real-data.md)，用 MIT/Stanford/TRI 公开数据集做电芯寿命早期预测，并把结果作为工具接入 Agent。
5. **Genie / Dashboard**：用 `battery_service_campaign` 做一个 AI/BI Dashboard（按车型/地区的风险地图、待处置工时），或建一个 Genie 空间让运营用自然语言提问“本周南方地区有多少 P0 车？”。

## 8.6 学完自测（面试常问）

1. Logged Model、Registered Model、Model Version、Alias 的区别？为什么下游用别名？
2. 类别不平衡（故障 12%）时应该看哪些指标？如何选阈值？
3. Feature Store 如何避免 training-serving skew？BMS 快照数据为什么需要 point-in-time join？
4. Serving 端点的 scale-to-zero 有什么利弊？实时车联网场景该怎么配置？
5. AI Gateway 能提供哪些能力？推理表的数据可以用来做什么？
6. 为什么质保判断、安全红线要放在 UC 函数里计算，而不是交给 LLM？
7. Agent 的 `resources` 声明的作用是什么？
8. 如何评估一个 Agent？LLM 裁判、Guidelines、自定义 scorer 各适合什么？安全指标的上线标准是什么？
9. Agent Bricks Knowledge Assistant 与代码 Agent 各适合什么场景？

## 8.7 清理资源 💰

运行 `99_cleanup.py`：
- 删除风险端点、Agent 端点（及可选的外部模型端点）；
- 可选：删除整个 `ev_battery_lab` schema（表、模型、函数、Volume、特征表）；
- Agent Bricks 创建的 Knowledge Assistant 需要在 **Agents** 页面手动删除；
- 定时 Job 记得暂停或删除。

- 做过阶段 9 的话，Volume `raw_battery_data` 中约 8 GB 原始文件可以删除（Delta 表已保留解析结果）。

🎉 恭喜完成整个 AI/ML 功能区实战！想挑战真实数据 → [阶段 9：真实数据](09-real-data.md)
