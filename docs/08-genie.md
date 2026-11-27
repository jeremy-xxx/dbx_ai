# 阶段 8：Genie 数据助手

> **目标**：让电池研发工程师用自然语言查询电芯数据和预测结果，不用每次都找数据团队写 SQL。
> **notebook**：`08_genie.py` + 界面操作 **耗时**：约 45 分钟

---

## 8.1 Genie 是什么
**Genie Agent**（旧称 Genie Space）是 Databricks 的数据问答助手：用户用自然语言提问，比如“哪种快充策略寿命最长？”，它把问题翻译成 SQL，执行后返回结果，并可以生成图表。这类“能查数据的 Agent”通常叫 **data agent**。

Genie 现在是一个产品家族（2026-11 官方文档）：

| 产品 | 给谁用 | 作用 |
|---|---|---|
| **Genie Agents** | 数据团队（你） | 配置可信的数据、业务说明、示例查询 |
| **Genie One** | 业务人员（工程师） | 统一的提问入口，答案基于 Genie Agents 配置的数据 |
| **Genie Code** | 开发者 | notebook 里的 AI 编程助手；创建 Genie Agent 时也会自动帮你补充说明 |

截至文档编写时，用户使用 Genie Agents **免费到 2027 年 1 月 31 日**（服务主体除外）。

**要求**：数据在 Unity Catalog 中；有 Pro 或 Serverless SQL Warehouse 的使用权限；workspace 开启了 Partner-powered AI features（由管理员控制）。

## 8.2 Genie 答得准，靠的是数据准备
Genie 的准确率**主要取决于你给它的数据和说明**，而不是模型本身。`08_genie.py` 做了三件事：

**1. 建宽表视图 `cell_overview`**
把元数据、关键特征、最新预测合成一张表，一行一个电芯。Genie 不用自己猜“预测表和元数据表怎么关联”“同一电芯有多次预测该取哪次”，出错的机会少得多。

**2. 给每一列写业务含义**
比如 `qc_decision` 的三个取值分别是什么意思、`charge_time_0_80_min` 越短代表什么。列名里没有的业务知识，都要写进说明。

**3. 准备示例问题和标准 SQL**
既用来教 Genie 怎么写这类查询，也作为检验 Genie 答案的标准答案。

## 8.3 创建 Genie Agent（界面操作）
1. 左侧 **Genie Agents → New**；
2. 数据源选 `main.battery_life_lab.cell_overview`。如果想回答“某个电芯第 50 个循环的容量是多少”这类逐循环的问题，可以再加上 `cell_cycle_summary` → **Create**；
3. 创建后 **Genie Code** 会自动读取数据，建议表说明和示例问题。逐条看一遍，接受合适的；
4. **Instructions**（业务说明）：粘贴 notebook 第 3 节打印的文字。其中最关键的是**业务术语的定义**，例如“短寿命 = 实际寿命低于 550”“需要复检 = qc_decision 为 short_life 或 borderline”；
5. **Example SQL queries**：添加 notebook 第 2 节的 1–2 个示例（问题 + SQL）；
6. 用 notebook 里的 4 个问题测试，和标准答案对比；
7. 满意后点 **Share** 分享给工程师。

官方还提供了进一步提升质量的方法，比如添加指标（metrics）、业务规则、已验证答案（verified answers），见官方文档 “Tune Genie Agent quality”。

## 8.4 检验 Genie 的答案
不要因为 Genie 给出了答案就直接相信。对每个测试问题：
1. 展开 Genie 回答里的 **SQL**，看它的筛选条件和关联方式对不对；
2. 和 notebook 第 2 节的标准答案对比结果。

最常见的错误是**理解错业务术语**，比如把“短寿命”理解成“预测为短寿命”而不是“实际短寿命”。这时要补充 Instructions，而不是换着说法反复问。

## 8.5 和其它方式的对比

| 方式 | 适合 |
|---|---|
| 写好的 SQL / Dashboard | 固定的、反复看的指标，比如每批的短寿命比例 |
| **Genie Agent** | 临时的、多变的问题，提问的人不会写 SQL |
| 自己写 Text-to-SQL 程序 | 需要嵌入自己的应用、完全控制流程 |

Genie Agent 也可以作为 **MCP 工具**接入其它 Agent（Playground → Tools → MCP Servers → Genie Agent），让一个更大的助手在需要查数据时调用它。

## 8.6 检查点
- [ ] Catalog 中出现视图 `cell_overview`，每列都有说明
- [ ] Genie Agent 对 4 个示例问题的回答和标准答案一致
- [ ] 做完练习 2，理解业务说明对 Genie 的影响

➡️ 可选：[阶段 9：实时 API](09-serving.md)；学完请运行 `99_cleanup.py`
