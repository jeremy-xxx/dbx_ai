# 附录：常见问题与排查

## A. 功能入口看不到

| 看不到 | 可能原因 | 处理 |
|---|---|---|
| 侧边栏没有 AI/ML 区域 | 工作区 persona 或权限设置 | 左上角切换到 “Machine Learning” 视图，或联系管理员 |
| Playground 没有 `databricks-` 模型 | 区域未开通 Foundation Model APIs / 被管理员禁用 | 联系管理员；或使用外部模型端点（阶段 5 选做） |
| Agents 页面没有 Agent Bricks 模板 | 区域/版本未开放、未启用 Serverless、Beta 功能未开启 | 管理员在 **Previews** 中启用；否则跳过 7.2，代码 Agent 不受影响 |
| AI Gateway 页面为空 | 还没有给任何端点配置网关 | 完成阶段 5 后再看 |
| Features 页面找不到特征表 | 表没有主键约束 | 必须通过 `fe.create_table` 或带 `PRIMARY KEY` 的建表语句创建 |

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
