# 阶段 9（可选）：部署成实时 API

> **目标**：了解当业务需要“随时调用”时，怎样把同一个模型部署成实时接口。
> **notebook**：`09_serving.py`（默认关闭，`RUN_SERVING = True` 才执行） **耗时**：约 1 小时
> **⚠️ 费用**：会创建在线特征存储和 Serving 端点两个持续计费的资源；Free Edition 可能不支持在线特征存储

---

## 9.1 什么时候需要
主线的批量预测已经满足“一批测完统一打分”。如果测试台软件希望**每个电芯一测完 100 个循环就立刻拿到预测**，比如用于自动分拣，就需要一个随时可调用的 API。

## 9.2 为什么还需要在线特征存储
模型是用 `fe.log_model` 保存的，调用时只传电芯 ID，所以端点要**实时查特征**：

```
测试台软件 ──{"cell_id": "b3c5"}──► Serving 端点 ──毫秒级查询──► 在线特征存储（Lakebase）
                                         │                              ▲
                                         ▼                              │ publish_table 同步
                                  返回 log10(寿命)               特征表 cell_early_features
```

| | 特征表（Delta） | 在线特征存储（Lakebase） |
|---|---|---|
| 适合 | 批量读取：训练、批量打分 | 按主键单条查询：实时服务 |
| 单次查询延迟 | 秒级 | 毫秒级 |
| 费用 | 只付存储 | 常驻数据库，持续计费 |

`publish_table` 把特征表同步到在线存储，前提是特征表开启了 Change Data Feed（阶段 3 已开启）。特征表更新后，再调用一次 `publish_table` 即可同步。

## 9.3 部署要点
- `scale_to_zero_enabled=True`：空闲约 30 分钟后缩到 0，不收算力费；代价是之后第一次调用要等几分钟冷启动；
- 端点名、实体名、模型名是三个不同的名字：

| 名字 | 代码 | 作用 |
|---|---|---|
| 端点名 | `SERVING_ENDPOINT` | 决定 API 地址，workspace 内唯一 |
| 实体名 | `ServedEntityInput(name="cell-life-champion")` | 端点内区分不同的模型版本（做 A/B 测试时用） |
| 模型名 + 版本 | `entity_name` + `entity_version` | 必须匹配 UC 中已注册的模型和版本 |

## 9.4 调用方式
notebook 里用 SDK 调用。在本地终端用 REST 调用的写法：

```bash
curl -X POST "https://<workspace>/serving-endpoints/<端点名>/invocations" -H "Authorization: Bearer $DATABRICKS_TOKEN" -H "Content-Type: application/json" -d '{"dataframe_records": [{"cell_id": "b3c5"}]}'
```
`DATABRICKS_TOKEN` 是你的个人访问令牌（PAT），在 Databricks 用户设置中生成。返回的是 log10(寿命)，需要自己换算：10 的这个数次方。

## 9.5 用完立即清理
运行 `99_cleanup.py`，它会删除 Serving 端点和在线特征存储。

## 9.6 检查点
- [ ] 端点状态 Ready，只传 `cell_id` 能拿到预测
- [ ] 能说出在线特征存储和特征表的区别
- [ ] 已运行 `99_cleanup.py`
