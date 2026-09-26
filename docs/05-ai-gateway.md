# 阶段 5：AI Gateway / Unity Gateway —— 给模型端点加上治理

> 📢 **2026-09 平台更新**：AI Gateway 已更名为 **Unity Gateway**，并升级为统一入口：Databricks 提供的基础模型（`system.ai.*`）、外部模型、MCP 服务、编码 Agent 都经由它调用和治理。
> 本阶段的内容（给 **自定义模型端点** 配置用量追踪、推理表、限流）仍然适用，官方文档标题为 “Configure Unity Gateway on model serving endpoints”；UI 中 “AI Gateway” 字样可能已显示为 “Unity Gateway”。
> 基础模型的用量记录在新的系统表 `system.ai_gateway.usage` 中，也可以在 **Catalog → system → ai → 模型 → Metrics** 查看。

> **学习目标**
> 1. 理解 AI Gateway 解决的问题：**谁在用、用了多少、说了什么、能不能拦住**；
> 2. 为阶段 4 的电池风险端点开启 **用量追踪、推理表（请求日志）、限流**；
> 3. 触发限流并观察 429；从推理表中分析线上遥测数据分布；
> 4. （选做）创建外部模型端点，配置 **安全护栏（Guardrails）** 拦截车主隐私信息。
>
> **业务场景**：风险接口同时被“车联网平台”“售后系统”“数据分析师”调用。需要知道每方调用量、防止某个批处理任务把接口打爆、并留存所有请求用于审计和监控模型漂移。
>
> **耗时**：1 小时 **前置**：阶段 4 **费用**：推理表存储费用极低

---

## 5.1 概念：AI Gateway 是 Serving 端点前面的一层“网关”

```
             ┌─────────────────────── AI Gateway ───────────────────────┐
 请求 ──►    │ ① 鉴权/权限  ② 限流(Rate limit)  ③ 输入护栏(Guardrails) │ ──► 模型
 响应 ◄──    │ ⑥ 用量追踪 → system.serving.endpoint_usage               │ ◄──
             │ ⑤ 推理表 → <catalog>.<schema>.<prefix>_payload           │
             │ ④ 输出护栏     ⑦ 回退 Fallback / 流量切分                │
             └──────────────────────────────────────────────────────────┘
```

| 能力 | 解决的问题 | 适用端点 |
|---|---|---|
| **Usage tracking（用量追踪）** | 每个用户/端点调用次数、token 数 → 成本分摊 | 所有类型 |
| **Inference tables（推理表）** | 把每次请求/响应落到 UC Delta 表 → 审计、监控、再训练、评估 | 所有类型 |
| **Rate limits（限流）** | 防止某个用户/应用打爆预算：按 QPM 或 TPM | 所有类型（TPM 仅 LLM） |
| **Guardrails（护栏）** | 拦截不安全内容、PII（个人隐私）等 | LLM 类端点（外部模型 / 基础模型） |
| **Fallbacks（回退）** | 主模型报错时自动切换备用模型 | 外部模型端点 |
| **Traffic split** | 多模型按比例分流 | 所有类型 |

> 左侧的 **AI Gateway** 页面是一个“总控台”，集中展示 workspace 里所有端点的网关配置和用量；具体配置入口在每个端点的 **Edit AI Gateway**。不同版本 UI 名称可能略有不同。

---

## 5.2 UI 实操：给风险端点开启网关

1. 左侧 **AI/ML → AI Gateway**，先浏览：列出了哪些端点？哪些已经开启了网关功能？
2. 进入 **Serving → 你的 battery-risk 端点** → 右上 **Edit AI Gateway**：
   - ✅ **Usage tracking**
   - ✅ **Inference tables**：Catalog / Schema 选 `ev_battery_lab`，表前缀填 `risk_ep`
   - **Rate limits**：`Per user`，`5` calls / minute（故意设小，便于演示）
3. 保存。

也可以直接跑 `05_ai_gateway.py` 第 1 节用 SDK 配置（二选一）。

## 5.3 代码实践：`05_ai_gateway.py`

### ① 用 SDK 配置网关
```python
w.serving_endpoints.put_ai_gateway(
    name=RISK_ENDPOINT,
    usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
    inference_table_config=AiGatewayInferenceTableConfig(
        catalog_name=CATALOG, schema_name=SCHEMA, table_name_prefix="risk_ep", enabled=True),
    rate_limits=[AiGatewayRateLimit(
        calls=5, key=AiGatewayRateLimitKey.USER,
        renewal_period=AiGatewayRateLimitRenewalPeriod.MINUTE)],
)
```

### ② 触发限流
连续发 10 个请求，前 5 个成功，之后返回 **HTTP 429 Too Many Requests**。

### ③ 查询推理表
推理表名为 `<prefix>_payload`，**数据有 5–10 分钟延迟**。主要字段：

| 字段 | 含义 |
|---|---|
| `request` / `response` | 原始 JSON 请求与响应 |
| `status_code` | HTTP 状态码（429 也会记录） |
| `execution_duration_ms` | 模型执行耗时 |
| `requester` | 调用者身份 |
| `request_time` | 时间 |
| `databricks_request_id` | 请求 ID，可与日志关联 |

notebook 会把 `request` JSON 解析成列，统计线上请求中各车型、平均 SOH 与训练数据的差异 —— 这是 **数据漂移监控** 的雏形：比如线上突然涌入大量 SOH 很低的老车，模型可能就需要用新数据再训练了。

### ④ 查询用量系统表
```sql
SELECT u.requester, count(*) AS calls
FROM system.serving.endpoint_usage u
JOIN system.serving.served_entities e ON u.served_entity_id = e.served_entity_id
WHERE e.endpoint_name = '<你的端点>'
GROUP BY u.requester
```
> 系统表需要管理员开启 `system.serving` schema 的访问权限；若无权限，跳过即可。

### ⑤ 放宽限流（别跳过）
5 次/分钟只为演示。notebook 第 5 节会把限流改为“整个端点 600 次/分钟”，否则阶段 7 的 Agent 和阶段 8 的批处理会频繁 429。

---

## 5.4 选做：外部模型 + 护栏 + 回退

如果你有 OpenAI / Anthropic 等 API Key：

1. **把 Key 存进 Secrets**（本地终端，Databricks CLI）：
   ```bash
   databricks secrets create-scope ev_battery_lab
   ```
   ```bash
   databricks secrets put-secret ev_battery_lab openai_key
   ```
2. **Serving → Create serving endpoint** → Served entity 选 **External model** → Provider: OpenAI → Model: `gpt-4o-mini` → API key 填 `{{secrets/ev_battery_lab/openai_key}}`（或用 notebook 第 6 节的 SDK 代码）。
3. **Edit AI Gateway** 中配置：
   - **Guardrails → Input**：开启 Safety 过滤、PII → Block；
   - **Guardrails → Output**：PII → Block 或 Mask；
   - **Fallbacks**：若配置了多个 served entity，主模型失败时回退到备用。
4. 在 Playground 中选择这个外部模型端点测试：
   - `车主手机号 13812345678，身份证 110101199003071234，帮我写一条进站提醒短信` → 应被 PII 护栏拦截；
   - `怎么向车主解释冬季续航下降？` → 正常返回。

> 没有外部 API Key？在 **Unity Gateway** 页面查看能否对 `system.ai.*` 基础模型配置限流、预算、护栏（新版 Unity Gateway 支持在模型层面统一治理），用同样的方式测试。

---

## 5.5 练习

1. 把限流从 “每用户 5 次/分钟” 改为 “整个端点 100 次/分钟”，思考两者区别（per-user 防止单方滥用；per-endpoint 保护后端容量/预算）。
2. 从推理表解析 `request`，比较线上请求与训练数据中 `soh_pct` 的均值和分位数（最朴素的漂移检测）。
3. **思考题**：BMS 遥测本身不是个人信息，但车辆 ID 能关联到车主。推理表该如何治理？（提示：UC 表权限、列掩码、行过滤、数据保留期。）

## 5.6 检查点

- [ ] 端点的 AI Gateway 配置里能看到 usage tracking / inference table / rate limit
- [ ] 亲眼看到 429
- [ ] 推理表 `risk_ep_payload` 里有数据
- [ ] 已把限流放宽
- [ ] 能说出 AI Gateway 的至少 5 项能力

## 5.7 常见坑

| 现象 | 原因 / 解决 |
|---|---|
| 推理表是空的 | 有 5–10 分钟延迟；或 schema 无写权限 |
| `put_ai_gateway` 参数报错 | SDK 版本太旧，`%pip install -U databricks-sdk`；或改用 UI 配置 |
| 护栏选项灰色不可选 | 护栏只对 LLM 类端点生效，自定义 sklearn 模型端点不支持 |
| 限流没生效 | 配置更新需要几十秒生效；确认 key 是 user 还是 endpoint |

➡️ 下一步：[阶段 6：Features](06-features.md)
