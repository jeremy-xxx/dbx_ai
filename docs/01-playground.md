# 阶段 1：Playground —— 零代码玩转大模型

> **学习目标**
> 1. 知道 Playground 能做什么、背后调用的是什么；
> 2. 会用 System Prompt、参数（temperature / max tokens）控制输出；
> 3. 会并排对比多个模型（以“车主反馈分类”为任务）；
> 4. 会把 Playground 里的对话“导出成代码”，在 notebook 里复现。
>
> **耗时**：45 分钟 **前置**：阶段 0 **费用**：按 token 计费，很少

---

## 1.1 概念：Playground 背后是什么？

Playground 本质上是一个 **聊天 UI**，背后调用的是 Databricks 上的模型服务。能选的模型分三类：

| 类别 | 例子 | 计费 | 在哪里看到 |
|---|---|---|---|
| **Databricks 提供的基础模型（按 token）** | `system.ai.claude-sonnet-4-5`、`system.ai.gpt-5-2` 等（旧版名称 `databricks-*`） | 按输入/输出 token | **Catalog → system → ai**（2026-09 起由 Unity Gateway 提供，不再列在 Serving 页面） |
| **External Models（外部模型）** | 你配置的 OpenAI / Anthropic / Gemini 端点 | 由外部厂商计费 | 你自己创建的 Serving 端点（阶段 5 选做） |
| **自定义 Agent / 模型端点** | 阶段 7 部署的电池售后 Agent | 按端点算力 | 你自己创建的 Serving 端点 |

> 🔑 **关键认知**：Playground 里做的一切，都能在代码里用 **OpenAI 兼容接口** 1:1 复现。这就是 Databricks “先 UI 原型、再代码工程化” 的思路。

---

## 1.2 UI 实操

### 练习 1：第一次对话
1. 左侧 **AI/ML → Playground**。
2. 左上角模型下拉框选择一个 Chat 模型（如 Claude Sonnet、GPT、Llama 等，具体列表因区域而异）。
3. 输入：`用通俗的语言解释什么是动力电池的 SOH（健康度），以及 LFP 和 NMC 电池在寿命上的区别。`
4. 观察回答下方显示的 **token 数与延迟**。

### 练习 2：System Prompt（系统提示词）
1. 点击对话框上方的 **System Prompt**（或设置里的 “System prompt”）。
2. 填入：
   ```
   你是一家电动车公司的资深电池售后工程师。回答必须：
   1) 使用简体中文；2) 不超过 150 字；3) 涉及安全风险时先给出安全提示；4) 最后给出一条可执行的建议。
   ```
3. 问：`车主说最近快充时电池包附近很烫，仪表还亮了黄灯，怎么回复？`
4. 去掉 System Prompt 再问一次，对比变化。

**思考**：System Prompt 和用户消息的区别是什么？（答：System Prompt 是对模型“角色与规则”的全局约束，每轮对话都生效。）

### 练习 3：参数调优
点开模型旁边的 **设置（⚙）**：

| 参数 | 含义 | 实验 |
|---|---|---|
| **Temperature** | 随机性，0=几乎确定，1+=更发散 | 设为 0 连问 3 次，再设为 1.2 连问 3 次，对比一致性 |
| **Max tokens** | 最大输出长度 | 设为 30，观察回答被截断 |
| **Top P** | 核采样 | 一般和 temperature 只调一个 |

> 对售后这种要求“口径一致”的场景，你会选择什么 temperature？

### 练习 4：多模型并排对比（结构化输出）
1. 点击右上角 **“+”**（Add model / Compare）添加第二、第三个模型。
2. 开启 **同步输入**（Sync），同一个问题同时发给多个模型。
3. 用下面这个 “车主反馈分类” 任务对比：
   ```
   请分析车主反馈，只输出 JSON：
   {"issue_type": "charging|range|thermal|warning_light|other",
    "safety_risk": "none|low|high",
    "needs_inspection": true|false,
    "reason": "..."}
   反馈："最近快充到 60% 以后就特别慢，而且电池包那里有点烫，仪表还亮了个黄灯。"
   ```
4. 再换两条反馈试试：
   - `冬天续航从 400 掉到 260 公里，正常吗？`
   - `车子用了三年，感觉比刚买的时候少跑了 30 公里。`
5. 记录：哪个模型 **JSON 最稳定**？哪个对 **安全风险判断最谨慎**？哪个 **最快 / 最便宜**？

### 练习 5：给模型挂工具（Tools）预览
Playground 支持给模型添加 **Tools**（Unity Catalog 函数、Vector Search 等），这正是 Agent 的雏形。
- 点 **Tools → Add tool**，添加内置的 `system.ai.python_exec`（让模型执行 Python）；
- 问：`一块 70 kWh 的电池，SOH 从 100% 降到 88%，按 4.5 km/kWh 算续航少了多少公里？请用 Python 计算。`
- 观察模型如何生成代码、调用工具、再根据结果回答。
- 阶段 7 我们会把自己写的 UC 函数（查车辆档案、查处置动作）挂上来，并 **一键导出为 Agent 代码**。

### 练习 6：导出代码
1. 在任意对话中点击 **Get code**（或 `</>` 图标）。
2. 你会看到 Python / curl / SQL (`ai_query`) 等多种调用方式。
3. 复制 Python 版本，对照 `notebooks/01_playground_to_code.py`，理解每一行。

---

## 1.3 代码实践：`01_playground_to_code.py`

| 小节 | 你会学到 |
|---|---|
| 1. 确认可用模型 | Catalog 中的 `system.ai.*`（gateway 模式）或旧版 `databricks-*` 端点 |
| 2. OpenAI 兼容客户端 | `get_llm_client()`（定义在 `00_config`），自动处理地址与鉴权 |
| 3. System prompt + temperature 实验 | 与 Playground 练习 2、3 对应 |
| 4. 结构化 JSON 输出：车主反馈分类 | `response_format` 与“解析失败兜底” |
| 5. MLflow Tracing 预告 | `mlflow.openai.autolog()`，每次调用自动记录到 **Experiments** |
| 6. SQL 里调大模型 | `ai_query()` —— 对整张 `vehicles` 表批量生成“电池状态点评” |

### 核心代码解读

```python
# Unity Gateway 方式（00_config 里 get_llm_client() 的实现）
from openai import OpenAI
w = WorkspaceClient()                                             # 在 notebook 里自动鉴权
token = w.config.authenticate()["Authorization"].split(" ", 1)[1]
client = OpenAI(api_key=token, base_url=f"{w.config.host}/ai-gateway/mlflow/v1")

resp = client.chat.completions.create(
    model="system.ai.claude-sonnet-4-5",          # ← UC 中的模型全名
    messages=[{"role": "system", "content": "..."},
              {"role": "user", "content": "..."}],
    temperature=0.2,
)
```

- 这是 **统一的 OpenAI 兼容接口**：换成 GPT、Gemini 等其他厂商的模型只需改 `model`，代码不变。所以 **任何支持 OpenAI SDK 的框架**（LangChain、LlamaIndex、DSPy…）都能直接接 Databricks 模型。
- 旧版方式：`w.serving_endpoints.get_open_ai_client()`，`base_url` 为 `https://<workspace>/serving-endpoints`，`model` 填 `databricks-*` 端点名。

```sql
SELECT vehicle_id,
       ai_query('system.ai.claude-sonnet-4-5',
                concat('用一句话点评这块电池的状态：', to_json(struct(*)))) AS comment
FROM vehicles LIMIT 5
```
> `ai_query` 调用 Unity Gateway 模型时，用量记录在系统表 `system.ai_gateway.usage` 中；限流、护栏等其他网关功能不作用于 `ai_query` 批处理。

- `ai_query` 让分析师在 SQL 里批量调用 LLM，阶段 8 会用它批量生成车主进站通知。

---

## 1.4 检查点

- [ ] 能说清 Playground 与 Serving 端点的关系
- [ ] 完成多模型对比，并记录了你心目中“结构化输出最稳”的模型
- [ ] notebook 中 `ai_query` 能跑通
- [ ] 在 **Experiments** 页面找到 `ev-battery-genai` 实验，并打开一条 **Trace**，看到了输入、输出、token 数

## 1.5 常见坑

| 现象 | 原因 / 解决 |
|---|---|
| Serving 页面找不到 `databricks-` 模型 | 正常：2026-09 起基础模型迁移到 Unity Gateway，去 **Catalog → system → ai** 查看 |
| 调用报 `model not found` / 404 | `LLM_MODEL_GATEWAY` 名称写错或该区域没有此模型；在 `system.ai` 里挑一个存在的；或改 `LLM_MODE = "legacy"` |
| 调用报 403 | 需要 workspace 的 Unity Gateway 查询权限（Workspace access 权限），联系管理员 |
| `ai_query` 报错不支持 | 需要 Serverless / SQL Warehouse / DBR 15.4+ |
| JSON 偶尔解析失败 | 降低 temperature；使用 `response_format`；代码中加 try/except 兜底 |
| 模型对“过热”判断不够谨慎 | 在 System Prompt 中明确安全规则——这正是阶段 7 要把规程交给 Agent 的原因 |

➡️ 下一步：[阶段 2：Experiments](02-experiments.md)
