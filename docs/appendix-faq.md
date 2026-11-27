# 附录：常见问题与排查

## A. 环境与导入

| 现象 | 处理 |
|---|---|
| 创建 Git folder 报 “The link to your GitHub account does not have access” | 在 GitHub 上给 Databricks GitHub App 授权访问这个仓库；或在 **Settings → Linked accounts** 改用个人访问令牌 |
| `%run ./00_config` 找不到文件 | `00_config` 必须和当前 notebook 在同一目录 |
| `%pip install` 后 import 的还是旧版本 | 确认运行了 `dbutils.library.restartPython()`，并且 `%run ./00_config` 在它之后 |
| 输出大量 `WARNING ... getContext().extraContext() is not a CommandContext accessor` | Serverless 下 MLflow 读不到某个环境标签而跳过，不影响结果，可忽略 |

## B. 数据接入（阶段 1）

| 现象 | 处理 |
|---|---|
| 下载很慢或中断 | 重跑会跳过已下载完的文件；`.part` 是未完成的文件 |
| workspace 不能访问外网 | 本地下载后用 `databricks fs cp` 上传到 Volume（见文档 1.3） |
| `AttributeError: 'Dataset' object has no attribute 'value'` | 新版 h5py 用 `ds[()]` 代替旧代码的 `.value` |
| 电芯数和 41 / 43 / 40 不一致 | 个别电芯寿命缺失被兜底规则剔除，少 1–2 个不影响学习 |

## C. 特征与训练（阶段 3–5）

| 现象 | 处理 |
|---|---|
| `MlflowException: The saved sklearn model references untrusted types` | 新版 MLflow 默认用 skops 格式保存模型。按报错里列出的类型加到 `skops_trusted_types`（项目已加 `numpy.dtype` 和 `sklearn.tree._tree.Tree`） |
| `fe.log_model` 提示 `artifact_path` 已弃用 | 只是提示，不影响使用 |
| 闸门没通过 | 回到 04 调整特征或模型；学习时可在 05 把 `FORCE_REGISTER` 改为 True |
| `get_model_version_by_alias` 报找不到 champion | 先成功运行 05 |

## D. 预测与监控（阶段 6–7）

| 现象 | 处理 |
|---|---|
| 06 提示“没有需要打分的电芯” | 这些电芯已被当前模型版本打过分，属于正常；想重新打分可先删除预测表中对应版本的记录 |
| 07 提示没有预测记录 | 先运行 06，且 widget `dataset_role` 为 `new_batch` |
| 新批次误差明显变大 | 这是本项目要演示的现象（分布漂移），见文档 7.4 的再训练闭环 |

## E. Genie（阶段 8）

| 现象 | 处理 |
|---|---|
| 左侧有 Genie Agents，但打不开 | workspace 未开启 Partner-powered AI features，需要账户管理员开启 |
| 创建时没有可选的 SQL Warehouse | Genie 需要 Pro 或 Serverless SQL Warehouse，并且你有 CAN USE 权限 |
| Genie 的答案和标准答案不一致 | 展开它的 SQL 看哪里理解错了，通常是业务术语，补充到 Instructions 中 |

## F. 实时 API（阶段 9）

| 现象 | 处理 |
|---|---|
| `create_online_store` 失败 | Free Edition 可能不支持在线特征存储；跳过阶段 9 不影响主线 |
| `publish_table` 报存储未就绪 | 在线存储创建需要几分钟，稍后重跑 |
| `TypeError: EndpointCoreConfigInput.__init__() missing ... 'name'` | 新版 databricks-sdk 要求 config 里也写 `name`（项目已修复） |
| 首次调用很慢或超时 | 端点开启了空闲缩到 0，冷启动需要几分钟 |

## G. 成本控制
- [ ] 学完运行 `99_cleanup.py`（删除 Serving 端点和在线特征存储）
- [ ] 暂停或删除阶段 6 创建的 Job
- [ ] 不再需要原始数据时，在 `99_cleanup.py` 中开启 `DELETE_RAW_FILES` 释放约 7.7 GB 存储

## H. 参考资料
- 论文：Severson et al., *Nature Energy* 4, 383–391 (2019)
- 数据：https://data.matr.io/1/ （CC BY 4.0）
- 官方加载代码：https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation
- Databricks 文档：Feature Engineering、MLflow 3、Genie Agents、Model Serving（`docs.databricks.com`，Azure / GCP 把路径中的 `aws` 换成 `azure` / `gcp`）
