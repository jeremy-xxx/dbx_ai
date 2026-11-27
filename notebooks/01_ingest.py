# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · 数据接入：下载 → Volume → 解析 → Delta 表
# MAGIC
# MAGIC 对应文档：`docs/01-ingest.md`
# MAGIC
# MAGIC **数据集**：Severson et al., *Data-driven prediction of battery cycle life before capacity degradation*, Nature Energy 4, 383–391 (2019)。
# MAGIC 124 个 A123 APR18650M1A（LFP/石墨，1.1 Ah）电芯，30℃ 恒温箱中以不同快充策略循环，直到容量衰减到 80%（0.88 Ah）。
# MAGIC 来源 https://data.matr.io/1/ ，**CC BY 4.0** 许可，使用时请引用上面的论文。
# MAGIC
# MAGIC ```
# MAGIC ① 下载 3 个批次的 .mat 文件（共 ~7.7 GB）到 UC Volume
# MAGIC ② 用 h5py 解析 MATLAB v7.3 结构体（只读需要的部分）
# MAGIC ③ 按官方加载代码清洗：剔除异常电芯、合并跨批次电芯的寿命标签
# MAGIC ④ 解析快充策略、划分数据集角色
# MAGIC ⑤ 写成 3 张 Delta 表：cell_metadata / cell_cycle_summary / cell_early_curves
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -U -q h5py requests

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## ① 下载到 UC Volume
# MAGIC - 三个文件共约 7.7 GB，下载需要 10–40 分钟（取决于网络）。已存在的文件会跳过，中断后重跑即可续上。
# MAGIC - workspace 无法访问外网时：本地下载后用 `databricks fs cp <本地文件> dbfs:/Volumes/<catalog>/battery_life_lab/raw_battery_data/` 上传（见文档 1.3）。

# COMMAND ----------

import os

import requests

BATCH_FILES = {
    "2017-05-12": ("https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download",
                   "2017-05-12_batchdata_updated_struct_errorcorrect.mat"),
    "2017-06-30": ("https://data.matr.io/1/api/v1/file/5c86bf13fa2ede00015ddd82/download",
                   "2017-06-30_batchdata_updated_struct_errorcorrect.mat"),
    "2018-04-12": ("https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download",
                   "2018-04-12_batchdata_updated_struct_errorcorrect.mat"),
}
BATCH_NO = {"2017-05-12": 1, "2017-06-30": 2, "2018-04-12": 3}

# 👀 界面：Catalog → main → battery_life_lab → Volumes 出现 raw_battery_data；下载完成后能看到 3 个 .mat 文件及大小
spark.sql(f"CREATE VOLUME IF NOT EXISTS {RAW_VOLUME} COMMENT '原始电池循环数据（Severson et al. 2019, CC BY 4.0）'")

for batch_date, (url, fname) in BATCH_FILES.items():
    dest = f"{RAW_VOLUME_PATH}/{fname}"
    if os.path.exists(dest) and os.path.getsize(dest) > 100 * 1024**2:
        print(f"已存在，跳过: {fname} ({os.path.getsize(dest) / 1024**3:.2f} GB)")
        continue
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=16 * 1024**2):   # 每次写 16 MB
                f.write(chunk)
                done += len(chunk)
                if done % (512 * 1024**2) < 16 * 1024**2:
                    print(f"  {fname}: {done / 1024**3:.2f} / {total / 1024**3:.2f} GB")
    os.replace(tmp, dest)   # 下载完整后才改名，避免留下半个文件
    print(f"✅ 下载完成: {fname}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ② 解析 MATLAB v7.3（HDF5）结构体
# MAGIC `.mat` v7.3 本质是 HDF5 文件。结构体里的字段以“对象引用”存储，所以要 `f[ref]` 二次取值。
# MAGIC 字段名参考官方仓库 `BuildPkl_Batch1.ipynb`。**只读需要的部分**，不把全部原始曲线读进内存：
# MAGIC
# MAGIC | 读取内容 | 用途 |
# MAGIC |---|---|
# MAGIC | `cycle_life`、`policy_readable` | 寿命标签、快充策略 |
# MAGIC | `summary`（每循环：放电/充电容量、内阻、温度、充电时间） | 容量衰减曲线、早期特征 |
# MAGIC | `cycles[9]` 与 `cycles[99]` 的 `Qdlin` | 第 10 / 100 循环按电压插值后的放电容量曲线（1000 点），用于核心特征 ΔQ(V) |

# COMMAND ----------

import h5py
import numpy as np
import pandas as pd

EARLY_IDX, LATE_IDX = 9, EARLY_CYCLES - 1   # 与官方 pkl 的键一致：cycles 的第 9 / 99 项（常称“第 10 / 100 循环”）
SUMMARY_FIELDS = {"QDischarge": "qd", "QCharge": "qc", "IR": "ir", "Tavg": "tavg",
                  "Tmin": "tmin", "Tmax": "tmax", "chargetime": "chargetime", "cycle": "cycle"}

meta_rows, cycle_frames, curve_rows = [], [], []
for batch_date, (_, fname) in BATCH_FILES.items():
    batch_no = BATCH_NO[batch_date]
    with h5py.File(f"{RAW_VOLUME_PATH}/{fname}", "r") as f:
        batch = f["batch"]
        n_cells = batch["summary"].shape[0]
        for i in range(n_cells):
            cell_id = f"b{batch_no}c{i}"
            cycle_life = float(np.ravel(f[batch["cycle_life"][i, 0]][()])[0])
            policy = f[batch["policy_readable"][i, 0]][()].tobytes()[::2].decode()   # MATLAB 字符串是 UTF-16

            summary = f[batch["summary"][i, 0]]
            s = {short: np.hstack(summary[name][0, :].tolist()) for name, short in SUMMARY_FIELDS.items()}
            cycle_frames.append(pd.DataFrame({"cell_id": cell_id, **s}))

            cycles = f[batch["cycles"][i, 0]]
            n_cycles = cycles["Qdlin"].shape[0]
            if n_cycles > LATE_IDX:
                q10 = np.hstack(f[cycles["Qdlin"][EARLY_IDX, 0]][()]).astype(float)
                q100 = np.hstack(f[cycles["Qdlin"][LATE_IDX, 0]][()]).astype(float)
                curve_rows.append((cell_id, q10.tolist(), q100.tolist()))

            meta_rows.append((cell_id, batch_no, batch_date, policy, cycle_life, n_cycles))
            if i % 10 == 0:
                print(f"  batch{batch_no} 解析进度 {i + 1}/{n_cells}")
    print(f"✅ {batch_date}: {n_cells} 个电芯")

meta = pd.DataFrame(meta_rows, columns=["cell_id", "batch", "batch_date", "charge_policy", "cycle_life", "n_cycles_recorded"])
cycles_pdf = pd.concat(cycle_frames, ignore_index=True)
curves = pd.DataFrame(curve_rows, columns=["cell_id", "qdlin_cycle10", "qdlin_cycle100"])
display(meta.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## ③ 按官方加载代码（`Load Data.ipynb`）清洗
# MAGIC 1. **batch1**：`b1c8/10/12/13/22` 未循环到 80% 容量就停止了 → 剔除；
# MAGIC 2. **跨批次电芯**：`b1c0–b1c4` 在 batch2 中继续测试（对应 `b2c7/8/9/15/16`）。
# MAGIC    我们只用前 100 个循环做特征，而这部分数据都在 batch1 里，所以 **只需把寿命标签加上续测的循环数**，再删除 batch2 中的这 5 条；
# MAGIC 3. **batch3**：`b3c2/23/32/37/42/43` 噪声大或提前终止 → 剔除；
# MAGIC 4. 兜底：剔除寿命为 NaN、或记录不足 100 个循环的电芯。

# COMMAND ----------

DROP = {"b1c8", "b1c10", "b1c12", "b1c13", "b1c22",
        "b3c2", "b3c23", "b3c32", "b3c37", "b3c42", "b3c43"}
CARRY_OVER = {"b1c0": ("b2c7", 662), "b1c1": ("b2c8", 981), "b1c2": ("b2c9", 1060),
              "b1c3": ("b2c15", 208), "b1c4": ("b2c16", 482)}

for b1, (b2, extra) in CARRY_OVER.items():
    meta.loc[meta.cell_id == b1, "cycle_life"] += extra
    DROP.add(b2)

meta = meta[~meta.cell_id.isin(DROP)]
meta = meta[meta.cycle_life.notna() & meta.cell_id.isin(curves.cell_id)].copy()
meta["cycle_life"] = meta.cycle_life.astype(int)
print("清洗后各批次电芯数:", meta.groupby("batch").size().to_dict(), "（官方口径 41 / 43 / 40）")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ④ 解析快充策略 + 划分数据集角色
# MAGIC
# MAGIC **快充策略** `5.4C(40%)-3.6C` 的含义：先以 5.4C 电流充到 40% 电量，再以 3.6C 充到 80%（之后统一 1C 充满）。
# MAGIC 解析成三个数值 `c1`、`q1`、`c2`，并算出“0→80% 充电时间”，方便分析快充对寿命的影响。
# MAGIC
# MAGIC **数据集角色**（模拟真实的研发流程）：
# MAGIC
# MAGIC | 角色 | 来源 | 用途 |
# MAGIC |---|---|---|
# MAGIC | `train` | batch1+2 交替选取（沿用论文划分） | 训练模型 |
# MAGIC | `test` | batch1+2 的另一半 | 评估模型、决定能否上线 |
# MAGIC | `new_batch` | batch3 全部（晚近一年开始的实验） | **模拟“上线后新到的一批电芯”**：阶段 6 给它们打分，阶段 7 等真实寿命出来后检验模型 |

# COMMAND ----------

parsed = meta.charge_policy.str.extract(r"([\d.]+)C\((\d+)%\)-([\d.]+)C")
meta["c1"] = pd.to_numeric(parsed[0], errors="coerce")   # 第一段充电倍率
meta["q1"] = pd.to_numeric(parsed[1], errors="coerce")   # 切换点（电量百分比）
meta["c2"] = pd.to_numeric(parsed[2], errors="coerce")   # 第二段充电倍率
# 0→80% 充电时间（分钟）：先以 c1 充 q1%，再以 c2 充到 80%
meta["charge_time_0_80_min"] = (60 * (meta.q1 / 100 / meta.c1 + (80 - meta.q1) / 100 / meta.c2)).round(1)
print("策略解析失败的电芯:", meta[meta.c1.isna()].charge_policy.tolist())

# 论文划分：batch1+2 按顺序交替（偶数位 test、奇数位 train，另外第 84 个进 test）；batch3 = new_batch
meta["_order"] = meta.cell_id.str.extract(r"b(\d)c(\d+)").astype(int).apply(tuple, axis=1)
meta = meta.sort_values("_order").drop(columns="_order").reset_index(drop=True)
is_b12 = meta.batch.isin([1, 2])
position = pd.Series(np.nan, index=meta.index)
position[is_b12] = np.arange(is_b12.sum())
meta["dataset_role"] = np.where(~is_b12, "new_batch",
                                np.where((position % 2 == 0) | (position == 83), "test", "train"))
print(meta.groupby(["batch", "dataset_role"]).size())
print("循环寿命范围:", meta.cycle_life.min(), "–", meta.cycle_life.max())

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⑤ 写入 Delta 表

# COMMAND ----------

keep = set(meta.cell_id)

# 👀 界面：Catalog → main → battery_life_lab → Tables 出现 cell_metadata / cell_cycle_summary / cell_early_curves
(spark.createDataFrame(meta)
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(CELL_META_TABLE))

cyc = cycles_pdf[cycles_pdf.cell_id.isin(keep)].copy()
cyc["cycle"] = cyc["cycle"].astype(int)
(spark.createDataFrame(cyc)
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(CELL_CYCLE_TABLE))

(spark.createDataFrame(curves[curves.cell_id.isin(keep)])
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(CELL_CURVE_TABLE))

# 表和列的注释：Catalog 里能看到，阶段 8 的 Genie 也会读取这些说明来理解数据
spark.sql(f"COMMENT ON TABLE {CELL_META_TABLE} IS '电芯元数据（Severson et al. 2019, CC BY 4.0）：批次、快充策略、循环寿命、数据集角色'")
spark.sql(f"COMMENT ON TABLE {CELL_CYCLE_TABLE} IS '每电芯每循环汇总：qd 放电容量(Ah)、qc 充电容量(Ah)、ir 内阻(Ω)、温度(℃)、chargetime 充电时间(min)'")
spark.sql(f"COMMENT ON TABLE {CELL_CURVE_TABLE} IS '第10/100循环按电压(3.6V→2.0V)线性插值的放电容量曲线，各1000点'")
for col, comment in {
    "cycle_life": "循环寿命：容量衰减到额定值 80%（0.88 Ah）时的循环次数",
    "charge_policy": "快充策略，格式 C1(Q1%)-C2：先以 C1 倍率充到 Q1% 电量，再以 C2 倍率充到 80%",
    "c1": "第一段充电倍率（C）", "q1": "两段充电的切换点（电量 %）", "c2": "第二段充电倍率（C）",
    "charge_time_0_80_min": "按策略计算的 0→80% 电量充电时间（分钟），越短表示快充越激进",
    "dataset_role": "train=训练集，test=测试集，new_batch=模拟上线后新到的一批电芯（batch3）",
}.items():
    spark.sql(f"ALTER TABLE {CELL_META_TABLE} ALTER COLUMN {col} COMMENT '{comment}'")

print(f"✅ {len(meta)} 个电芯、{len(cyc)} 行循环数据已写入")
