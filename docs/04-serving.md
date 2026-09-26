# 阶段 4：Serving —— 把电池风险模型变成 REST API

> **学习目标**
> 1. 理解 Model Serving 的三种端点类型；
> 2. 用 UI 和 SDK 两种方式部署 `battery_risk_model@champion`；
> 3. 用 SDK、REST（requests / curl）、SQL `ai_query` 三种方式调用端点；
> 4. 掌握 scale-to-zero、工作负载大小、流量切分（A/B）等配置；
> 5. 会看端点日志与指标排查问题。
>
> **业务场景**：车辆每次回传 BMS 数据时，车联网平台实时调用这个接口，拿到故障概率，高风险车自动在 App 里提示车主预约检测。
>
> **耗时**：1 小时（含等待部署 10–20 分钟） **前置**：阶段 3 **费用**：⚠️ 端点运行时计费，务必开启 scale-to-zero

---

## 4.1 概念：三种 Serving 端点

| 类型 | 服务对象 | 例子 | 计费 |
|---|---|---|---|
| **Custom Model Serving** | 你在 UC 注册的 MLflow 模型（sklearn、pyfunc、Agent…） | 本阶段的 `battery_risk_model` | 按 CPU/GPU 算力时长 |
| **基础模型（Foundation Models）** | Databricks 提供的开源/商业大模型 | `system.ai.claude-sonnet-4-5`（旧版名 `databricks-*`） | Pay-per-token 或预留容量。**2026-09 起由 Unity Gateway 管理，不再显示在 Serving 页面** |
| **External Models** | 代理到外部厂商（OpenAI、Anthropic、Bedrock、Vertex…） | 阶段 5 选做 | 外部厂商计费 |

三者**统一被 Unity Gateway（原 AI Gateway）治理**。2026-09 起 Serving 页面主要承载 **你自己部署的模型**（本阶段的风险模型、阶段 7 的 Agent），基础模型的入口移到了 Unity Gateway。

```
车联网平台 ──HTTPS──► https://<workspace>/serving-endpoints/<endpoint>/invocations
                          │
                     [AI Gateway：鉴权/限流/日志/护栏]   ← 阶段 5
                          │
               ┌──────────┴──────────┐
        served entity A (90%)   served entity B (10%)   ← 流量切分
        battery_risk_model v2   battery_risk_model v1
```

**关键配置**

| 配置 | 说明 | 建议 |
|---|---|---|
| **Served entity** | 端点背后的模型版本（一个端点可以挂多个） | 引用具体版本号，上线新版本时更新端点配置 |
| **Workload size** | Small（0–4 并发）/ Medium / Large | 学习用 Small |
| **Scale to zero** | 无流量约 30 分钟后缩到 0，不收算力费 | ✅ 学习时必开；缺点是冷启动要几分钟 |
| **Traffic split** | 多个 served entity 之间按百分比分流 | 用于 A/B 测试、金丝雀发布 |
| **Environment variables** | 可引用 Secrets：`{{secrets/scope/key}}` | 放 API Key 等 |

> 💭 **思考**：真实车联网场景下，能接受 scale-to-zero 的冷启动吗？（不能：实时告警需要常驻实例，生产中应关闭 scale-to-zero 并设置最小并发。学习阶段为省钱开启。）

---

## 4.2 方式一：UI 部署（推荐先做一遍）

1. 左侧 **AI/ML → Serving → Create serving endpoint**。
2. **Name**：填 `00_config` 打印出来的 `RISK_ENDPOINT`（如 `zhangsan-battery-risk`）。
3. **Served entities**：
   - Entity：选择 **Unity Catalog** 中的 `ev_battery_lab.battery_risk_model`，Version 选 champion 对应的版本；
   - Compute type：CPU；Workload size：Small；
   - ✅ 勾选 **Scale to zero**。
4. （先不要管 AI Gateway 区域，阶段 5 再配。）
5. **Create** → 状态从 `Not Ready` 变为 `Ready`，需要 10–20 分钟。
6. 等待期间，点 **Logs → Build logs** 看镜像构建过程（安装 requirements.txt 中的依赖）。

> 如果你已经用 UI 创建了，notebook 中的创建代码会检测到并改为“更新配置”，不会冲突。

## 4.3 方式二：SDK 部署（`04_model_serving.py`）

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()
served = [ServedEntityInput(
    entity_name=BASE_MODEL_NAME, entity_version=champion_version,
    workload_size="Small", scale_to_zero_enabled=True)]
w.serving_endpoints.create_and_wait(name=RISK_ENDPOINT,
    config=EndpointCoreConfigInput(served_entities=served), timeout=timedelta(minutes=40))
```

## 4.4 调用端点的 4 种方式

**① SDK**
```python
resp = w.serving_endpoints.query(name=RISK_ENDPOINT, dataframe_records=[{...12 个特征...}])
resp.predictions   # [[0.12, 0.88]]  → 故障概率 88%
```

**② Python requests（任何语言同理）**
```python
requests.post(f"{host}/serving-endpoints/{RISK_ENDPOINT}/invocations",
              headers=w.config.authenticate(),            # {"Authorization": "Bearer ..."}
              json={"dataframe_records": records})
```

**③ curl（本地终端，需要 Personal Access Token）**
```bash
curl -X POST "https://<workspace-host>/serving-endpoints/<endpoint>/invocations" -H "Authorization: Bearer $DATABRICKS_TOKEN" -H "Content-Type: application/json" -d '{"dataframe_records":[{"vehicle_age_months":60,"odometer_km":380000,"cycle_count":1200,"fast_charge_ratio":0.75,"avg_depth_of_discharge":0.85,"max_cell_temp_c_30d":58,"cell_voltage_imbalance_mv":130,"bms_alerts_90d":5,"soh_pct":72,"vehicle_model":"commercial_van","battery_chemistry":"NMC","climate_zone":"hot"}]}'
```

**④ SQL `ai_query`**
```sql
SELECT vehicle_id,
       ai_query('<endpoint>', named_struct('vehicle_age_months', CAST(vehicle_age_months AS DOUBLE), ...),
                returnType => 'ARRAY<DOUBLE>')[1] AS fault_proba
FROM vehicles LIMIT 10
```

请求体支持的格式：`dataframe_records`（行列表，最常用）、`dataframe_split`（columns + data）、`inputs`（张量）。

---

## 4.5 UI 观察

打开端点页面：
- **Metrics**：QPS、延迟（P50/P99）、错误率、CPU/内存、并发数；
- **Logs → Service logs**：模型运行时 `print` 与报错；
- **Query endpoint**（右上角按钮）：直接在浏览器发请求测试，也会显示 curl/Python 示例；
- **Served entities**：看当前版本、状态、scale-to-zero 设置。

## 4.6 练习

1. **What-if 分析**：固定一辆车的其它特征，只把 `fast_charge_ratio` 从 0.1 调到 0.9（步长 0.2），调用端点画出故障概率曲线。再对 `max_cell_temp_c_30d` 做同样的事。这是向业务方解释模型的好方法。
2. **流量切分（金丝雀发布）**：在 UI **Edit endpoint** 中再添加一个 served entity（旧版本），设置 90% / 10% 流量。连续请求 20 次观察分布。练完改回单版本。
3. **冷启动**：等端点缩到 0 后再请求，记录首次延迟。
4. **错误请求**：故意少传一个特征或传错类型（`"soh_pct": "abc"`），观察 schema 校验报错信息。
5. **上线新版本**：训练一个新模型、注册、移动 champion，然后用 `update_config_and_wait` 更新端点。

## 4.7 检查点

- [ ] 端点状态 `Ready`
- [ ] 至少用 2 种方式成功调用
- [ ] scale-to-zero 已开启
- [ ] 完成 What-if 分析，能讲清“快充占比 → 风险”的关系

## 4.8 常见坑

| 现象 | 原因 / 解决 |
|---|---|
| 部署卡在 `Updating` 很久然后失败 | 看 **Build logs**：通常是依赖安装失败；检查 `requirements.txt` |
| `403 Forbidden` | Token 无该端点 `CAN QUERY` 权限 |
| `Model is missing inputs ['xxx']` | 请求缺少签名中的列 |
| 首次请求超时 | scale-to-zero 冷启动，等待后重试 |
| Free Edition 无法创建 | 配额限制，删除其他端点后重试 |

➡️ 下一步：[阶段 5：AI Gateway](05-ai-gateway.md)
