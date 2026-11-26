# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · AI Gateway (legacy)：给自部署端点配置用量追踪、推理表、限流、护栏
# MAGIC
# MAGIC > 官方现在把“端点上的网关配置”称为 **AI Gateway (legacy)**，仍可使用；新版统一治理平台叫 **Unity Gateway**（左侧菜单 AI Gateway），见文档 5.4.1。
# MAGIC
# MAGIC 对应文档：`docs/05-ai-gateway.md`

# COMMAND ----------

# MAGIC %pip install -U -q databricks-sdk openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayInferenceTableConfig,
    AiGatewayRateLimit,
    AiGatewayRateLimitKey,
    AiGatewayRateLimitRenewalPeriod,
    AiGatewayUsageTrackingConfig,
)

w = WorkspaceClient()
INFERENCE_PREFIX = "risk_ep"
PAYLOAD_TABLE = f"{CATALOG}.{SCHEMA}.{INFERENCE_PREFIX}_payload"

def gateway_config(calls: int, key: AiGatewayRateLimitKey) -> dict:
    return dict(
        usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
        inference_table_config=AiGatewayInferenceTableConfig(
            catalog_name=CATALOG, schema_name=SCHEMA, table_name_prefix=INFERENCE_PREFIX, enabled=True),
        rate_limits=[AiGatewayRateLimit(
            calls=calls, key=key, renewal_period=AiGatewayRateLimitRenewalPeriod.MINUTE)],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 配置 AI Gateway：用量追踪 + 推理表 + 每用户 5 次/分钟
# MAGIC 如果你已经在 UI 里配过，这里会覆盖为相同的配置。

# COMMAND ----------

# 👀 界面：Serving → 端点页面的 AI Gateway 区域显示已开启 Usage tracking、Inference tables、Rate limits；
#          （左侧 AI Gateway 菜单进入的是新版 Unity Gateway 控制台，主要管理基础模型和 MCP，不一定显示这个端点）
w.serving_endpoints.put_ai_gateway(name=RISK_ENDPOINT, **gateway_config(5, AiGatewayRateLimitKey.USER))

print(w.serving_endpoints.get(RISK_ENDPOINT).ai_gateway)
print("⏳ 等待 60 秒让配置生效 …")
time.sleep(60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 触发限流：连续 10 次请求
# MAGIC 预期：前 5 次 200，之后 429。

# COMMAND ----------

import requests

url = f"{w.config.host}/serving-endpoints/{RISK_ENDPOINT}/invocations"
headers = {**w.config.authenticate(), "Content-Type": "application/json"}

sample = (spark.table(VEHICLES_TABLE).select("vehicle_id", *MODEL_FEATURES).limit(10).toPandas()
          .astype({c: "float64" for c in NUMERIC_FEATURES}))

# 👀 界面：端点页面 → Metrics 中 4xx 错误数上升（被限流的 429 请求）
for i, row in enumerate(sample.to_dict(orient="records"), 1):
    vid = row.pop("vehicle_id")
    r = requests.post(url, headers=headers, json={"dataframe_records": [row]}, timeout=120)
    detail = f"故障概率 {r.json()['predictions'][0][1]:.1%}" if r.ok else r.text[:120]
    print(f"#{i:>2} {vid}  HTTP {r.status_code}  {detail}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 查询推理表
# MAGIC ⚠️ 数据有延迟（官方说明不到一小时）。如果表不存在或为空，稍后再运行本单元。

# COMMAND ----------

# 👀 界面：Catalog → main → ev_battery_lab → Tables 出现 risk_ep_payload（首次请求后有延迟，官方说明不到一小时）
if spark.catalog.tableExists(PAYLOAD_TABLE):
    display(spark.sql(f"""
        SELECT request_time, status_code, execution_duration_ms, requester, request, response
        FROM {PAYLOAD_TABLE}
        ORDER BY request_time DESC
        LIMIT 20
    """))
else:
    print(f"{PAYLOAD_TABLE} 还不存在，过几分钟再试")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 把 request JSON 解析成列 → 线上 vs 训练数据分布（漂移监控雏形）

# COMMAND ----------

if spark.catalog.tableExists(PAYLOAD_TABLE):
    display(spark.sql(f"""
        WITH online AS (
          SELECT explode(from_json(request:dataframe_records, 'ARRAY<MAP<STRING, STRING>>')) AS rec
          FROM {PAYLOAD_TABLE}
          WHERE status_code = 200
        )
        SELECT 'online' AS source, count(*) AS n,
               round(avg(CAST(rec['soh_pct'] AS DOUBLE)), 2)            AS avg_soh,
               round(avg(CAST(rec['fast_charge_ratio'] AS DOUBLE)), 3)  AS avg_fast_charge
        FROM online
        UNION ALL
        SELECT 'training', count(*), round(avg(soh_pct), 2), round(avg(fast_charge_ratio), 3)
        FROM {VEHICLES_TABLE}
    """))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 用量系统表（默认仅账户管理员可查询；Free Edition 下通常无权限，会自动跳过）

# COMMAND ----------

try:
    display(spark.sql(f"""
        SELECT date_trunc('HOUR', u.request_time) AS hour, u.requester, u.status_code, count(*) AS calls
        FROM system.serving.endpoint_usage u
        JOIN system.serving.served_entities e ON u.served_entity_id = e.served_entity_id
        WHERE e.endpoint_name = '{RISK_ENDPOINT}'
        GROUP BY ALL ORDER BY hour DESC
    """))
except Exception as e:
    print("无法访问系统表（需要管理员授权），跳过：", str(e)[:200])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 放宽限流（重要！）
# MAGIC 5 次/分钟只是为了演示。阶段 7 的 Agent 和阶段 8 的批处理都会调用这个端点，
# MAGIC 做完上面的实验后把限流改成宽松值，否则后面会频繁遇到 429。

# COMMAND ----------

# 👀 界面：端点 AI Gateway 区域的 Rate limits 变为“整个端点 600 次/分钟”
w.serving_endpoints.put_ai_gateway(name=RISK_ENDPOINT, **gateway_config(600, AiGatewayRateLimitKey.ENDPOINT))
print("✅ 限流已放宽为整个端点 600 次/分钟")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.（选做）外部模型端点 + 护栏
# MAGIC 前提：已按文档 5.4 把 OpenAI Key 存进 secret scope `ev_battery_lab`，key 名 `openai_key`。
# MAGIC 把下面的 `RUN_OPTIONAL` 改成 True 再运行。
# MAGIC
# MAGIC > 护栏相关字段在不同 SDK 版本中可能调整；若报错，请在 UI 中配置，效果相同。

# COMMAND ----------

RUN_OPTIONAL = False
EXTERNAL_ENDPOINT = RISK_ENDPOINT.replace("battery-risk", "ext-gpt")

if RUN_OPTIONAL:
    from datetime import timedelta
    from databricks.sdk.service.serving import (
        AiGatewayGuardrailParameters, AiGatewayGuardrailPiiBehavior,
        AiGatewayGuardrailPiiBehaviorBehavior, AiGatewayGuardrails,
        EndpointCoreConfigInput, ExternalModel, ExternalModelProvider,
        OpenAiConfig, ServedEntityInput,
    )

    # 👀 界面：Serving 页面出现 <你的用户名>-ext-gpt 外部模型端点
    w.serving_endpoints.create_and_wait(
        name=EXTERNAL_ENDPOINT,
        config=EndpointCoreConfigInput(name=EXTERNAL_ENDPOINT, served_entities=[ServedEntityInput(
            name="gpt-4o-mini",
            external_model=ExternalModel(
                name="gpt-4o-mini",
                provider=ExternalModelProvider.OPENAI,
                task="llm/v1/chat",
                openai_config=OpenAiConfig(openai_api_key="{{secrets/ev_battery_lab/openai_key}}"),
            ),
        )]),
        timeout=timedelta(minutes=20),
    )

    # 👀 界面：该端点 AI Gateway 区域显示 Guardrails（护栏）配置
    w.serving_endpoints.put_ai_gateway(
        name=EXTERNAL_ENDPOINT,
        usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
        guardrails=AiGatewayGuardrails(
            input=AiGatewayGuardrailParameters(
                safety=True,
                pii=AiGatewayGuardrailPiiBehavior(behavior=AiGatewayGuardrailPiiBehaviorBehavior.BLOCK),
            ),
            output=AiGatewayGuardrailParameters(
                pii=AiGatewayGuardrailPiiBehavior(behavior=AiGatewayGuardrailPiiBehaviorBehavior.BLOCK),
            ),
        ),
    )

    client = w.serving_endpoints.get_open_ai_client()
    for text in ["怎么向车主解释冬季续航下降？",
                 "车主手机号 13812345678，身份证 110101199003071234，帮我写一条进站提醒短信"]:
        try:
            r = client.chat.completions.create(model=EXTERNAL_ENDPOINT, messages=[{"role": "user", "content": text}])
            print("✅", text, "→", r.choices[0].message.content[:100])
        except Exception as e:
            print("🛡️ 被拦截:", text, "→", str(e)[:200])
