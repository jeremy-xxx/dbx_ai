# 附录：常见问题与排查

## A. 功能入口看不到

| 看不到 | 可能原因 | 处理 |
|---|---|---|
| 侧边栏没有 AI/ML 区域 | 工作区 persona 或权限设置 | 左上角切换到 “Machine Learning” 视图，或联系管理员 |
| Serving 页面没有 `databricks-` 基础模型端点 | **2026-09 起基础模型迁移到 Unity Gateway**（页面顶部有横幅提示） | 去 **Catalog → system → ai** 查看模型，`00_config` 用 `LLM_MODE = "gateway"`；需要旧端点可点横幅中的 legacy experience，并设 `LLM_MODE = "legacy"` |
| Catalog 里也没有 `system.ai` 模型 | 区域不支持 Unity Gateway / 无查询权限 / 被管理员禁用 | 联系管理员；或使用外部模型端点（阶段 5 选做） |
| Agents 页面没有 Agent Bricks 模板 | 区域/版本未开放、未启用 Serverless、Beta 功能未开启 | 管理员在 **Previews** 中启用；否则跳过 7.2，代码 Agent 不受影响 |
| AI Gateway 页面为空 | 还没有给任何端点配置网关 | 完成阶段 5 后再看 |
| Features 页面找不到特征表 | 表没有主键约束 | 必须通过 `fe.create_table` 或带 `PRIMARY KEY` 的建表语句创建 |

### 模型可用性测试
不确定该用哪种模式、哪个模型时，新建 notebook 运行（不依赖 `00_config`）：

```python
%pip install -q openai
```
```python
dbutils.library.restartPython()
```
```python
from databricks.sdk import WorkspaceClient
from openai import OpenAI

w = WorkspaceClient()
legacy = w.serving_endpoints.get_open_ai_client()
token = w.config.authenticate()["Authorization"].split(" ", 1)[1]
gateway = OpenAI(api_key=token, base_url=f"{w.config.host.rstrip('/')}/ai-gateway/mlflow/v1")

for mode, client, model in [
    ("legacy",  legacy,  "databricks-meta-llama-3-3-70b-instruct"),
    ("legacy",  legacy,  "databricks-claude-haiku-4-5"),
    ("gateway", gateway, "system.ai.claude-sonnet-4-5"),
]:
    try:
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "用一句话介绍你自己"}], max_tokens=50)
        print(f"✅ [{mode}] {model} → {r.choices[0].message.content}")
    except Exception as e:
        print(f"❌ [{mode}] {model} → {str(e)[:120]}")
```

**Free Edition 实测结果（2026-09）**：Llama 3.3 70B（legacy）✅；Claude 系列 404；gpt-oss 能调用但返回推理块列表而非纯文本（本项目代码按纯文本解析，不建议选）。

| 现象 | 说明 |
|---|---|
| `DeprecationWarning: get_open_ai_client() is deprecated` | 只是提示，功能正常。官方建议今后改用 `databricks-openai` 包的 `DatabricksOpenAI`，本项目为兼容性暂保留 |
| 返回内容是 `[{'type': 'reasoning', ...}]` 而不是文字 | 推理类模型（如 gpt-oss）的返回格式，换 Llama 等非推理模型即可 |

## B. 环境与依赖

**Q：Serverless 和经典 ML Runtime 集群选哪个？**
Serverless 启动快、免运维，本项目全部 notebook 都按 Serverless 编写（开头 `%pip install` 所需包）。用经典集群时选 **ML Runtime 15.4 LTS 及以上**。

**Q：`%pip install` 之后 import 还是旧版本？**
必须执行 `dbutils.library.restartPython()`（每个 notebook 已包含），且 `%run ./00_config` 要放在重启之后。

**Q：`%run ./00_config` 报找不到文件？**
`%run` 使用相对路径，要求 `00_config` 与当前 notebook 在同一目录。

**Q：`07a` 报“没找到规程文档”？**
需要保持目录结构：`notebooks/` 与 `data/` 是兄弟目录。用 Git folder 导入最省事。

## C. MLflow 相关

| 报错 | 处理 |
|---|---|
| `log_model() got an unexpected keyword argument 'name'` | MLflow 2.x：改为 `artifact_path="model"`，或升级到 MLflow 3 |
| `Model signature is required for UC` | log_model 时传 `signature` 或 `input_example` |
| `RESOURCE_DOES_NOT_EXIST: alias champion` | 先跑阶段 3 设置别名 |
| `register_model` 注册到了 Workspace Registry | 加 `mlflow.set_registry_uri("databricks-uc")`，模型名用三级命名 |

## D. Serving 相关

| 现象 | 处理 |
|---|---|
| 端点一直 `Not Ready` / 构建失败 | 端点页面 → **Logs → Build logs**，多数是依赖安装失败 |
| 首次请求很慢或超时 | scale-to-zero 冷启动，等待 1–5 分钟后重试 |
| `429 Too Many Requests` | 阶段 5 限流生效，运行 `05` 第 5 节放宽 |
| `ai_query` 调自定义模型报类型错误 | 加 `returnType => 'ARRAY<DOUBLE>'`，并把整型列 `CAST` 成 double |
| 端点数量达到上限 | 删除不用的端点（`99_cleanup`） |

## E. Agent 相关

| 现象 | 处理 |
|---|---|
| `get_vehicle_profile` 总是返回 NULL | 函数 join 了阶段 6 的特征表 `battery_features`，先运行 `06_feature_engineering` |
| 红线车没有被升级为 P0 处置 | 检查 System Prompt 中的“安全红线优先”规则；换 tool calling 更强的模型；用评估中的 `safety_redline_escalation` 回归测试 |
| 本地 `AGENT.predict` 正常，部署后工具 403 | `resources` 漏声明（端点、函数、表），重新 log + deploy |
| LLM 不调用工具或参数乱填 | 换 tool calling 能力更强的模型；改进函数 COMMENT 与 system prompt；降低 temperature |
| `DatabricksFunctionClient` 执行函数报错 | 确认安装 `unitycatalog-ai[databricks]`；确认你对函数有 `EXECUTE` 权限；首次调用需要等待 serverless 启动 |
| `ModelConfig` 找不到 `agent_config.yaml` | 先运行 `07b` 第 ① 节生成配置文件；`agent.py` 与 yaml 需在同一目录 |
| 评估时 LLM 裁判报错 | 内置裁判使用 Databricks 托管的评估模型，需要 workspace 开启相关功能（Partner-powered AI features） |
| Review App 打不开 | 端点需 Ready；访问者需要对应权限 |

## F. 权限速查

| 操作 | 需要的权限 |
|---|---|
| 建表、建函数、建 Volume | schema 上 `CREATE TABLE` / `CREATE FUNCTION` / `CREATE VOLUME` |
| 注册模型 | schema 上 `CREATE MODEL` |
| 加载模型 / 部署 | 模型上 `EXECUTE` |
| 调用端点 | 端点上 `CAN QUERY` |
| 修改端点 / AI Gateway | 端点上 `CAN MANAGE` |
| 查询系统表 | 管理员授予 `system.serving` 的 `SELECT` |

## G. 成本控制清单

- [ ] 所有自定义端点开启 **scale-to-zero**
- [ ] 学习期间不用 Provisioned Throughput（按小时计费）
- [ ] `ai_query` 批量调用前先 `LIMIT` 小样本测试
- [ ] 学完运行 `99_cleanup.py`，并在 Agents 页面删除 Knowledge Assistant
- [ ] 暂停/删除阶段 8 创建的定时 Job

## H. 参考文档（官方，以最新为准）

- MLflow 3 on Databricks：`docs.databricks.com/aws/en/mlflow/`
- Model Serving：`docs.databricks.com/aws/en/machine-learning/model-serving/`
- AI Gateway：`docs.databricks.com/aws/en/ai-gateway/`
- Feature Engineering in Unity Catalog：`docs.databricks.com/aws/en/machine-learning/feature-store/`
- Agent Framework & Agent Bricks：`docs.databricks.com/aws/en/generative-ai/agent-framework/`、`.../generative-ai/agent-bricks/`
- MLflow GenAI 评估：`docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/`

（Azure / GCP 把路径中的 `aws` 换成 `azure` / `gcp`。）
