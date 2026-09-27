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
| **Databricks 提供的基础模型（按 token）** | `system.ai.qwen3-next-80b-a3b-instruct`、`system.ai.llama-4-maverick` 等（可用列表见 Unity Gateway → Models） | 按输入/输出 token | **Unity Gateway → Models**（2026-09 起不再列在 Serving 页面） |
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
点开 Playground 左侧的 **设置（⚙）**，会看到下面这些参数（界面默认值见“默认”列）：

| 参数 | 默认 | 含义 |
|---|---|---|
| **Temperature** | On，1 | **随机性**。0 = 每次几乎选概率最高的词，回答稳定；越大越发散、越有创意 |
| **Top P** | Off（0.95） | **核采样**：只从累计概率达到 P 的候选词里挑。0.95 = 去掉最不可能的 5% 候选 |
| **Top K** | Off（1） | 只从概率最高的 K 个候选词里挑。K = 1 等于每次都选第一名，结果完全确定 |
| **Max tokens** | 5000 | 回答的**最大长度**（token 数）。超过就截断，也是控制成本的主要手段 |
| **Add other parameter** | — | 手动添加其它参数，输入参数名后回车，例如 `stop`（遇到指定文字就停止）、`seed`（随机种子，让结果可复现） |

**开关（On / Off）的含义**：Off 表示**不发送这个参数**，由模型使用自己的默认值；打开后才会把你设的数值发给模型。Get code 导出的代码里，也只会包含处于 On 状态的参数。

**实验**：
1. **Temperature = 0**，同一个问题连问 3 次；再拉到 **1**（滑块最右）连问 3 次，对比回答的一致性；
2. **Max tokens = 30**，观察回答被截断（阶段 1 notebook 里通过代码的 `max_tokens` 参数做同样的事）；
3. 关掉 Temperature，打开 **Top K = 1**，连问 3 次，看结果是不是和 Temperature = 0 一样稳定。

> 💡 Temperature、Top P、Top K 都是在控制“从候选词里怎么挑”。实际使用中**一般只调 Temperature**，其它保持 Off，同时调多个会让效果难以预测。
>
> 对售后这种要求“口径一致”的场景，你会选择什么 temperature？（参考：阶段 7 的 Agent 用的是 0.1）

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
Playground 支持给模型添加工具，这正是 Agent 的雏形。点模型名称右侧的 **Tools**，打开 **Add tools** 窗口，有 4 个标签页：

| 标签 | 作用 | 本项目 |
|---|---|---|
| **UC Function** | 逐个挑选 Unity Catalog 函数当工具（如 `system.ai.python_exec`、阶段 7 自建的函数） | ✅ 阶段 1、7 主要用这个 |
| **Function Definition** | 只写工具的 JSON 说明书（名称、参数），**不含执行代码**，用来测试模型会不会调用、传什么参数 | 可选 |
| **Vector Search** | 挂向量索引，让模型检索文档（RAG） | 未使用 |
| **MCP Servers** | 通过 MCP 协议一次接入一组工具（见下表） | 了解即可 |

**MCP Servers** 页里的 5 种来源（MCP = Model Context Protocol，标准化的工具接入协议，任何支持 MCP 的 Agent 都能用）：

| 来源 | 接入什么 |
|---|---|
| **UC MCP Service** | Databricks 内置的外部服务：`web_search`、`github`、`gmail`、`slack` 等（请求经 Unity Gateway 转发；gmail 这类需要先授权账号） |
| **Unity Catalog Function** | 选一个 **schema**，把其中**所有**函数整批作为工具（与 UC Function 标签的区别：按 schema 整批 vs 逐个挑选） |
| **AI Search** | 向量索引的 MCP 形式（Vector Search 在新版中叫 AI Search） |
| **Genie Agent** | 把 Genie 空间（自然语言查表助手）作为工具 |
| **Custom MCP Server** | 自己在 Databricks Apps 上部署的 MCP 服务（App 名须以 `mcp-` 开头） |

**实验**（模型选 Qwen3 Next Instruct）：
1. **Tools → UC Function** 标签，添加 `system.ai.python_exec`（让模型执行 Python），保存；
2. 问：`一块 70 kWh 的电池，SOH 从 100% 降到 88%，按 4.5 km/kWh 算续航少了多少公里？请用 Python 计算。`
3. 观察模型如何生成代码、调用工具、再根据结果回答；
4. （可选）**MCP Servers → UC MCP Service** 选 `web_search`，问一个需要联网的问题，比如 “2026 年动力电池行业有什么新闻？”。

阶段 7 我们会把自己写的 UC 函数（查车辆档案、查处置动作）挂上来，并通过 **Get code → Create agent notebook** 导出为 Agent 代码。

### 练习 6：导出代码（Get code 菜单）
Playground 顶部模型名称右侧有 **Get code** 下拉菜单，可以把你调好的对话“导出”：

| 选项 | 生成什么 | 何时用 |
|---|---|---|
| **Python API** | Python 调用代码（`OpenAI(api_key=..., base_url=".../ai-gateway/mlflow/v1")`） | **本阶段** |
| **Curl API** | 命令行调用示例 | 本阶段 |
| **Apply on data** | 对一张表批量调用模型的 `ai_query()` SQL | **本阶段**，阶段 8 也会用到 |
| **Create agent notebook** | 包含 Agent 代码的 notebook（MLflow `ResponsesAgent` + 工具 + 注册/部署到 Model Serving） | 阶段 7 |
| **Export to Databricks Apps**（Recommended） | 把 Agent 导出为 Databricks App 项目（官方推荐的新部署方式） | 阶段 7（了解即可） |

> ⚠️ **挂了工具后，Python API / Curl API 会变灰**：它们只能导出“调用一次模型”的代码，无法复现“模型要求调用工具 → 执行工具 → 结果交回模型”这个循环（练习 5 里 Playground 替你完成了工具执行）。带工具的对话要用 **Create agent notebook** 或 **Export to Databricks Apps** 导出完整 Agent。想导出简单调用代码，先在 **Tools** 里移除工具。

本阶段练习：
1. 选择 **Qwen3 Next Instruct**（**不挂工具**），发一句话，点 **Get code → Python API**，对照 `notebooks/01_playground_to_code.py` 和 `00_config.py` 里的 `get_llm_client()`，理解每一行。区别：导出代码从环境变量读令牌（适合在 Databricks 外部运行），项目代码自动使用你的登录身份；
2. 点 **Get code → Apply on data**，选择 `vehicles` 表，生成一段 `ai_query` SQL，和 notebook 第 6 节手写的 SQL 对比。

---

## 1.3 代码实践：`01_playground_to_code.py`

| 小节 | 你会学到 |
|---|---|
| 1. 确认可用模型 | Unity Gateway → Models 中的 `system.ai.*` 模型服务 |
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
    model="system.ai.qwen3-next-80b-a3b-instruct",          # ← UC 中的模型全名
    messages=[{"role": "system", "content": "..."},
              {"role": "user", "content": "..."}],
    temperature=0.2,
)
```

> 📍 **在 Databricks 界面里确认地址**：左侧 **AI Gateway**（页面标题显示为 Unity Gateway）→ **Models** → 点任意模型 → **Overview → Get started**，里面有 cURL / Python 示例，地址就是 `https://<你的workspace>/ai-gateway/mlflow/v1/...`，模型名是 `system.ai.<服务名>`。
> - 示例用的是 `/responses`（OpenAI Responses 接口），本项目用的是 `/chat/completions`（聊天接口），两者都在 `/ai-gateway/mlflow/v1` 下；
> - 页面上的 **Generate Access Token** 用于在 Databricks **外部**（如本地电脑）调用；在 notebook 里用 `get_llm_client()` 会自动使用你的登录身份，不需要生成令牌。

- 这是 **统一的 OpenAI 兼容接口**：换成 GPT、Gemini 等其他厂商的模型只需改 `model`，代码不变。所以 **任何支持 OpenAI SDK 的框架**（LangChain、LlamaIndex、DSPy…）都能直接接 Databricks 模型。
```sql
SELECT vehicle_id,
       ai_query('system.ai.qwen3-next-80b-a3b-instruct',
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
| 调用报 `model not found` / 404 | `LLM_MODEL` 名称写错或该区域没有此模型；以 **Unity Gateway → Models** 页面列出的模型为准（Catalog 的 Models 登记册里的名字不一定能调用） |
| 调用报 403 | 需要 workspace 的 Unity Gateway 查询权限（Workspace access 权限），联系管理员 |
| `ai_query` 报错不支持 | 需要 Serverless / SQL Warehouse / DBR 15.4+ |
| JSON 偶尔解析失败 | 降低 temperature；使用 `response_format`；代码中加 try/except 兜底 |
| 模型对“过热”判断不够谨慎 | 在 System Prompt 中明确安全规则——这正是阶段 7 要把规程交给 Agent 的原因 |

➡️ 下一步：[阶段 2：Experiments](02-experiments.md)
