# 阶段 7：Agents —— 构建、评估、部署电池售后运维 Agent

> **学习目标**
> 1. 理解 Agent 的本质：**LLM + 工具 + 循环**；
> 2. 用 **Agent Bricks**（低代码）基于维保/质保文档搭建 Knowledge Assistant；
> 3. 用 **UC 函数** 制作受治理的工具，并在 Playground 里挂载测试；
> 4. 用 **MLflow ResponsesAgent** 手写一个 tool-calling Agent，会看 Trace 调试；
> 5. 用 `mlflow.genai.evaluate` + LLM 裁判做 **系统化评估**（尤其是安全规则）；
> 6. 用 `agents.deploy` 部署，并通过 **Review App** 收集人工反馈。
>
> **耗时**：2–3 小时 **前置**：阶段 4（风险端点运行中）、阶段 6（特征表） **费用**：⚠️ Agent 端点 + LLM token

---

## 7.1 概念：Agent 到底是什么？

```
                 ┌──────────────────────── Agent 循环 ─────────────────────────┐
 服务顾问：       │ LLM：先查档案 → tool_call: get_vehicle_profile                │
 V01234 电池      │       ↓ UC 函数返回：safety_redline=true, in_warranty=true    │
 怎么了？该怎么办？│ LLM：评估风险 → tool_call: predict_battery_risk               │
 ────────────►   │       ↓ 查特征 + 调用阶段4的风险端点 → high (0.87)            │
                 │ LLM：看规程 → get_policy_doc('battery_maintenance_sop')        │
                 │ LLM：查动作 → get_maintenance_actions('commercial_van','high') │
                 │ LLM：看质保 → get_policy_doc('battery_warranty_policy')        │
 ◄────────────   │ LLM：不再需要工具 → 输出 P0 处置建议 + 车队沟通话术             │
 结构化建议       └─────────────────────────────────────────────────────────────┘
```

Databricks 上构建 Agent 的两条路：

| | **Agent Bricks**（侧边栏 Agents 页面） | **Agent Framework**（代码） |
|---|---|---|
| 方式 | 选模板、填数据源、点按钮 | Python（MLflow ResponsesAgent / LangGraph / OpenAI SDK…） |
| 典型模板 | Knowledge Assistant（文档问答）、Information Extraction（信息抽取）、Custom LLM（文本任务）、Multi-Agent Supervisor（多 Agent 编排） | 任意逻辑 |
| 优化 | 自动评估 + 自动优化 | 自己写评估、自己迭代 |
| 适合 | 标准场景快速落地 | 复杂业务逻辑、需要调用 ML 模型、需要完全掌控 |

两者最终都是 **Serving 端点**，都可以用 **MLflow Tracing** 追踪、用端点上的 **AI Gateway (legacy)** 配置治理，也可以互相组合。

> Agent Bricks 的可用性取决于区域和 workspace 设置（需要 Serverless、部分功能为 Beta）。如果你的 Agents 页面没有这些模板，直接跳到 7.3。

---

## 7.2 Part A：Agent Bricks —— Knowledge Assistant（低代码 RAG）

**先运行 `07a_agent_tools.py`**，它会把 `data/policies/*.md` 上传到 Volume `ev_battery_lab.policy_docs`。

1. 左侧 **AI/ML → Agents** → 选择 **Knowledge Assistant** → **Build**。
2. 填写：
   - **Name**：`ev-battery-policy-assistant`
   - **Description**：`回答售后人员关于电池维保规程、质保政策、充电常见问题的提问`
   - **Knowledge source**：类型选 **UC Files / Volume**，路径选 `/Volumes/<catalog>/ev_battery_lab/policy_docs/`，描述写“电池售后规程与质保文档”
   - **Instructions**（可选）：`用中文回答；引用条款编号；涉及安全的问题优先给出安全提示；不知道就说不知道`
3. **Create agent**，等待索引构建。系统自动完成：文档解析 → 切块 → 向量化 → 建向量索引 → 部署端点。
4. 在右侧测试面板提问：
   - `电芯压差多少算安全红线？触发后要怎么处理？`
   - `商用车的电池质保是几年多少公里？`
   - `SOH 75% 在质保期内能免费换电池吗？`
   - `LFP 电池可以每天充到 100% 吗？`
5. 观察回答中的 **引用（citations）**，点击跳到原文。
6. **Improve quality**：添加几条“问题 + 期望答案”作为标注，Agent Bricks 会据此评估并优化。

> 🔑 这就是 RAG（检索增强生成）：模型本身不知道你们公司的规程，靠“先检索、再回答”。代码版 Agent 里我们用了简化方案（按文档名整篇返回），文档一多就必须上向量检索。**阶段 7c 会亲手实现 RAG 的每一步**，见 [07c-rag.md](07c-rag.md)。
>
> **Knowledge Assistant 的局限**：它只会读文档，**不会查某辆车的数据、不会调用风险模型**。所以“V01234 该怎么办？”这种问题需要 Part B 的代码 Agent。

---

## 7.3 Part B-1：UC 函数工具（`07a_agent_tools.py`）

| 函数 | 作用 | 实现要点 |
|---|---|---|
| `get_vehicle_profile(vid)` | 车辆档案 + 最新电池状态 | join `vehicles` 与 `battery_features`（取最新遥测）；**在 SQL 中确定性计算** `in_warranty`、`soh_below_warranty_threshold`、`safety_redline` |
| `get_battery_features(vid)` | 风险模型的 12 个特征 | 读特征表，字段与模型签名一致 |
| `get_maintenance_actions(model_name, risk)` | 标准处置动作 | 读 `maintenance_actions` 表 |
| `get_policy_doc(doc)` | 规程全文 | 读 `policy_texts` 表 |

**写好工具的要点**：
1. **函数和参数的 COMMENT 就是给 LLM 看的说明书**。写清格式（“例如 V00042”）、返回什么、何时使用。
2. 返回 **结构化 JSON 字符串**，LLM 更容易理解。
3. **规则明确的判断（质保、红线）放在工具里算**，不要指望 LLM 做“96 个月且 15 万公里以内”这种比较——LLM 会算错，而这类错误在售后场景代价很高。
4. 查不到时返回 NULL / 明确提示，不要抛异常。
5. 参数名不要和表列名相同（避免 SQL 歧义，所以用 `vid` / `doc` 而不是 `vehicle_id` / `doc_name`）。

**Playground 练习**（重要！）：按 `07a` 最后一节的步骤，在 Playground 里通过 **Tools → UC Function** 标签逐个挂上这些 UC 函数聊天，然后点 **Get code → Create agent notebook** 导出 Agent 代码（**Export to Databricks Apps** 则会导出为 Databricks App）。

---

## 7.4 Part B-2：手写 Agent（`agent.py`）

我们手写了一个**最小但完整**的 Agent，让你看清每个环节：

```python
class BatteryServiceAgent(ResponsesAgent):          # MLflow 的 Agent 接口（兼容 OpenAI Responses API）
    def predict(self, request):
        messages = [system_prompt, *用户消息]
        for _ in range(MAX_STEPS):                  # ① 循环
            msg = llm.chat.completions.create(model=LLM_MODEL, messages=messages, tools=TOOLS)
            if not msg.tool_calls:                  # ② 没有工具调用 → 结束
                return 最终回答
            for tc in msg.tool_calls:               # ③ 执行工具，结果以 role=tool 回填
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": self.call_tool(tc.function.name, args)})
```

关键设计：

| 设计 | 说明 |
|---|---|
| `ResponsesAgent` | MLflow 3 推荐的 Agent 接口，部署后自动兼容 Review App、Playground、OpenAI 客户端 |
| `ModelConfig` | 端点名、函数名等放在 `agent_config.yaml`，部署时通过 `model_config` 注入 |
| `@mlflow.trace(span_type=TOOL)` + `mlflow.openai.autolog()` | 每次工具调用和 LLM 调用都生成 Trace span |
| `predict_battery_risk` | **混合工具**：先调 UC 函数取特征，再调阶段 4 的 ML 端点 —— 生成式 AI 与传统 ML 结合 |
| System Prompt 中的“安全红线优先” | 业务规则的优先级：确定性规则 > 模型概率 > LLM 判断 |
| 工具异常 → 返回错误文本 | 让 LLM 知道出错了并自行应对，而不是整个请求崩溃 |
| `MAX_STEPS` | 防止无限循环烧 token |
| `mlflow.models.set_model(AGENT)` | “models from code”：部署时直接执行这个文件，而不是 pickle 对象 |

> 生产中你也可以用 LangGraph（`databricks-langchain`）、OpenAI Agents SDK 等框架，只要包一层 `ResponsesAgent` 就能用同样的方式部署。

---

## 7.5 Part B-3：测试与调试（`07b` 第 ② 节）

`07b` 自动挑出三类典型车辆：**红线车**、**严重衰减车**（SOH < 72%）、**健康车**。运行后单元格下方出现 **Trace UI**：
- 左侧是调用树：`predict` → `ChatCompletion` → `call_tool(get_vehicle_profile)` → `ChatCompletion` → …
- 点任意节点看 **输入 / 输出 / 耗时 / token**。

**调试练习**：
1. 故意把 `get_vehicle_profile` 的 COMMENT 改得很模糊（如 “查车”），观察 LLM 是否还能正确使用。
2. 问一个不存在的车辆 `V99999`，看 Agent 是否编造信息。
3. 问“比较 V00001 和 V00002 哪台电池更危险”——观察 LLM 是否 **并行调用** 多个工具。
4. 把 System Prompt 里“安全红线优先”那条删掉，再问红线车，看结果是否变差——理解 prompt 中业务规则的作用。

---

## 7.6 Part B-4：评估（`07b` 第 ③ 节）

```python
mlflow.genai.evaluate(data=eval_data, predict_fn=predict_fn, scorers=[
    RelevanceToQuery(), Safety(),                                          # 内置 LLM 裁判
    Guidelines(name="safety_redline_escalation", guidelines="触发红线必须建议停止快充并24小时内进站"),
    Guidelines(name="pack_replacement_approval", guidelines="整包更换必须说明需区域技术经理审批"),
    Guidelines(name="no_fabrication", guidelines="查不到的车辆不能编造"),
    Guidelines(name="no_panic_wording", guidelines="话术不得使用恐慌性措辞"),
    no_probability_in_owner_script,                                        # 自定义代码评分器
])
```

| 评分器类型 | 适合检查 | 例子 |
|---|---|---|
| 内置裁判 | 通用质量 | 是否切题、是否安全、是否有依据（RAG 场景用 `RetrievalGroundedness`） |
| `Guidelines` | 自然语言描述的业务规则 | 红线升级、审批要求、不编造 |
| `@scorer` 自定义 | 可以用代码精确判断的规则 | 正则检查、JSON 格式、调用了哪些工具 |

结果在 **Experiments → ev-battery-genai → Evaluations** 中查看，每条都有裁判理由。

**评估驱动开发循环**：
```
改 prompt / 工具 / 模型 → 跑评估 → 对比指标 → 看失败样例的 Trace → 再改
```
> ⚠️ 安全相关指标目标是 **100%**。任何一条红线车没被升级处理，都不能上线。

---

## 7.7 Part B-5：注册与部署（`07b` 第 ④–⑧ 节）

1. **log_model**：`python_model="agent.py"` + `resources=[...]`。
   - `resources` 声明 Agent 需要访问的端点、函数、表；部署时 Databricks 自动为 Agent 创建一个有这些权限的短期凭证（**自动鉴权透传**）。
2. **mlflow.models.predict(env_manager="uv")**：在干净环境中试跑，提前发现依赖问题。
3. **register_model**：注册到 UC，`battery_service_agent`。
4. **agents.deploy**：创建端点 + 推理表 + Trace 监控 + **Review App**。

> ⚠️ **Unity Gateway 与部署方式（2026-09 更新）**
> - LLM 是 Unity Gateway 中的 `system.ai.*` 模型服务而不是 serving 端点，所以 `resources` 中不声明它。
> - 部署到 Model Serving 后，若 Agent 调用 LLM 报 401/403，说明自动鉴权透传的凭证无权访问 Unity Gateway。官方新推荐是把 Agent 部署到 **Databricks Apps**（文档 “Author an agent and deploy it on Databricks Apps”），Apps 支持更完整的鉴权范围（包括 `ai-gateway` 用户 API scope）。Model Serving 部署方式仍受支持，本项目为了与前面阶段衔接继续使用它。
> - 本地测试与评估（`07b` 的 ②③）用的是你本人的凭证，不受影响。

部署完成后：
- **Serving** 页面出现 `agents_<catalog>-ev_battery_lab-battery_service_agent` 端点；
- 打开 **Review App** 链接，像用 ChatGPT 一样对话，还可以对回答 👍/👎、写评语 —— 这些反馈会进入 MLflow，可转成评估集；
- **Playground** 下拉框里也能选到你的 Agent 端点，可以和基础模型并排对比；
- 在 Serving → 这个 Agent 端点的 **AI Gateway** 区域加上限流（复习阶段 5）。

---

## 7.8 练习（进阶）

1. **加工具**：新增 UC 函数 `get_fleet_summary(owner)`，返回某车队所有车辆的风险分布，让 Agent 能回答“顺达物流车队哪些车要优先进站？”。
2. **多 Agent**：在 **Agents → Multi-Agent Supervisor** 中，把 7.2 的 Knowledge Assistant 和 7.7 的代码 Agent 作为两个子 Agent，让 Supervisor 自动路由：“规程/质保/充电知识问题”→ Knowledge Assistant，“具体车辆分析”→ 代码 Agent。
3. **Information Extraction**：用 Agent Bricks 的 Information Extraction 模板，从一批“维修工单文本”中抽取 `故障部件 / 故障现象 / 处理方式 / 是否质保`。
4. **换框架**：用 `databricks-langchain` + LangGraph 的 `create_react_agent` 重写 `agent.py`，保持 `ResponsesAgent` 接口不变。
5. **生产监控**：为已部署的 Agent 配置生产监控（scheduled scorers），让 `safety_redline_escalation` 自动对线上 Trace 采样打分。

6. **MCP 工具**：在 Playground 的 **Tools → MCP Servers** 中试两种接入方式：
   - **Unity Catalog Function**：选 schema `ev_battery_lab`，把 4 个函数整批接入，观察模型是否会误用 `get_battery_features`（理解“逐个挑选”与“整批接入”的取舍）；
   - **UC MCP Service → web_search**：让 Agent 在给建议前联网查询某款电池的公开召回信息。思考：外部搜索结果可信吗？该如何在 System Prompt 中约束它的使用？

## 7.9 检查点

- [ ] （若可用）Knowledge Assistant 能带引用地回答规程问题
- [ ] Playground 里挂 UC 函数工具成功对话
- [ ] 红线车的回答中出现了 P0 处置（停止快充、24 小时内进站）
- [ ] 评估结果中 `safety_redline_escalation` 为 100%
- [ ] Agent 端点 Ready，Review App 可访问

## 7.10 常见坑

| 现象 | 原因 / 解决 |
|---|---|
| LLM 不调用工具，直接瞎答 | 换一个 tool calling 更强的模型；强化 system prompt；改进工具描述 |
| `工具调用失败: 429` | 阶段 5 的限流没放宽，运行 `05` 第 5 节 |
| `工具调用失败: ...PERMISSION_DENIED` | 部署后出现 → `resources` 漏声明；本地出现 → 你本人缺权限 |
| `get_vehicle_profile` 返回 NULL | 阶段 6 特征表未创建（函数 join 了特征表） |
| `mlflow.models.predict` 报缺包 | 把缺的包加进 `pip_requirements` |
| `agents.deploy` 很久没好 | 正常需要 ~15 分钟；看 Serving 页面该端点的 Build logs |
| `ResponsesAgent` 导入失败 | MLflow 版本 < 3.0，升级 `mlflow[databricks]>=3.1` |

➡️ 下一步：[阶段 7c：手写 RAG](07c-rag.md)
