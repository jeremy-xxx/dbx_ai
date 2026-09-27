# Databricks notebook source
# MAGIC %md
# MAGIC # 09a · 真实数据接入：MIT / Stanford / TRI 电芯循环寿命数据集
# MAGIC
# MAGIC 对应文档：`docs/09-real-data.md`
# MAGIC
# MAGIC **数据集**：Severson et al., *Data-driven prediction of battery cycle life before capacity degradation*, Nature Energy 4, 383–391 (2019)。
# MAGIC 124 个 A123 APR18650M1A（LFP/石墨，1.1 Ah）电芯，在 30℃ 恒温箱中以不同快充策略循环至容量衰减到 80%（0.88 Ah）。
# MAGIC 数据来源 https://data.matr.io/1/ ，**CC BY 4.0** 许可，使用时请引用上面的论文。
# MAGIC
# MAGIC 本 notebook：
# MAGIC ```
# MAGIC ① 下载 3 个批次的 .mat 文件（共 ~7.7 GB）到 UC Volume
# MAGIC ② 用 h5py 解析 MATLAB v7.3 结构体（只取需要的部分）
# MAGIC ③ 按官方加载代码的规则清洗（剔除异常电芯、合并跨批次电芯的寿命标签）
# MAGIC ④ 写成 3 张 Delta 表：cell_metadata / cell_cycle_summary / cell_early_curves
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
# MAGIC - 三个文件共约 7.7 GB，下载需要 10–40 分钟（取决于网络）。已存在的文件会跳过。
# MAGIC - 如果 workspace 无法访问外网：先在本地下载，再用 `databricks fs cp <本地文件> dbfs:/Volumes/<catalog>/<schema>/raw_battery_data/` 上传（见文档 9.3）。
# MAGIC - 想先快速试跑：widget `batches` 只填 `2017-06-30`（1.87 GB）。

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
dbutils.widgets.text("batches", ",".join(BATCH_FILES), "要处理的批次（逗号分隔）")
BATCHES = [b.strip() for b in dbutils.widgets.get("batches").split(",") if b.strip()]

# 👀 界面：Catalog → main → ev_battery_lab → Volumes 出现 raw_battery_data；下载完成后能看到 .mat 文件及大小
spark.sql(f"CREATE VOLUME IF NOT EXISTS {RAW_VOLUME} COMMENT '原始电池循环数据（Severson et al. 2019, CC BY 4.0）'")

def download(url: str, dest: str, chunk_mb: int = 16) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 100 * 1024**2:
        print(f"已存在，跳过: {dest} ({os.path.getsize(dest) / 1024**3:.2f} GB)")
        return
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_mb * 1024**2):
                f.write(chunk)
                done += len(chunk)
                if done % (512 * 1024**2) < chunk_mb * 1024**2:
                    print(f"  {os.path.basename(dest)}: {done / 1024**3:.2f} / {total / 1024**3:.2f} GB")
    os.replace(tmp, dest)
    print(f"✅ 下载完成: {dest}")

for b in BATCHES:
    url, fname = BATCH_FILES[b]
    download(url, f"{RAW_VOLUME_PATH}/{fname}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ② 解析 MATLAB v7.3（HDF5）结构体
# MAGIC `.mat` v7.3 本质是 HDF5 文件，结构体里的字段以“对象引用”存储，所以要 `f[ref]` 二次取值。
# MAGIC 字段名参考官方仓库 `BuildPkl_Batch1.ipynb`。我们 **只读取需要的部分**，不把全部原始曲线读进内存：
# MAGIC
# MAGIC | 读取内容 | 用途 |
# MAGIC |---|---|
# MAGIC | `cycle_life`、`policy_readable` | 标签、充电策略 |
# MAGIC | `summary`（每循环：放电/充电容量、内阻、温度、充电时间） | 容量衰减曲线、早期特征 |
# MAGIC | `cycles[9]` 与 `cycles[99]` 的 `Qdlin` | 第 10 / 100 循环按电压插值后的放电容量曲线（1000 点），用于核心特征 ΔQ(V) |

# COMMAND ----------

import h5py
import numpy as np
import pandas as pd

# 与官方 pkl 的键一致：cycles 字典的第 9 / 99 项（常被称为“第 10 / 100 循环”）
EARLY_IDX, LATE_IDX = 9, 99
SUMMARY_FIELDS = {"QDischarge": "qd", "QCharge": "qc", "IR": "ir", "Tavg": "tavg",
                  "Tmin": "tmin", "Tmax": "tmax", "chargetime": "chargetime", "cycle": "cycle"}

def parse_batch(path: str, batch_no: int):
    meta, cycle_rows, curves = [], [], []
    with h5py.File(path, "r") as f:
        batch = f["batch"]
        n = batch["summary"].shape[0]
        for i in range(n):
            cell_id = f"b{batch_no}c{i}"
            cycle_life = float(np.ravel(f[batch["cycle_life"][i, 0]][()])[0])
            policy = f[batch["policy_readable"][i, 0]][()].tobytes()[::2].decode()

            summ = f[batch["summary"][i, 0]]
            s = {short: np.hstack(summ[name][0, :].tolist()) for name, short in SUMMARY_FIELDS.items()}
            cycle_rows.append(pd.DataFrame({"cell_id": cell_id, **s}))

            cycles = f[batch["cycles"][i, 0]]
            n_cycles = cycles["Qdlin"].shape[0]
            if n_cycles > LATE_IDX:
                q10 = np.hstack(f[cycles["Qdlin"][EARLY_IDX, 0]][()]).astype(float)
                q100 = np.hstack(f[cycles["Qdlin"][LATE_IDX, 0]][()]).astype(float)
                curves.append((cell_id, q10.tolist(), q100.tolist()))

            meta.append((cell_id, batch_no, policy, cycle_life, n_cycles))
            if i % 10 == 0:
                print(f"  batch{batch_no} 解析进度 {i + 1}/{n}")
    return meta, cycle_rows, curves

BATCH_NO = {"2017-05-12": 1, "2017-06-30": 2, "2018-04-12": 3}
all_meta, all_cycles, all_curves = [], [], []
for b in BATCHES:
    m, c, q = parse_batch(f"{RAW_VOLUME_PATH}/{BATCH_FILES[b][1]}", BATCH_NO[b])
    all_meta += m; all_cycles += c; all_curves += q
    print(f"✅ {b}: {len(m)} 个电芯")

meta_pdf = pd.DataFrame(all_meta, columns=["cell_id", "batch", "charge_policy", "cycle_life", "n_cycles_recorded"])
cycles_pdf = pd.concat(all_cycles, ignore_index=True)
curves_pdf = pd.DataFrame(all_curves, columns=["cell_id", "qdlin_cycle10", "qdlin_cycle100"])
display(meta_pdf.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## ③ 按官方加载代码（`Load Data.ipynb`）的规则清洗
# MAGIC 1. **batch1**：`b1c8/10/12/13/22` 未循环到 80% 容量就停止了 → 剔除；
# MAGIC 2. **跨批次电芯**：`b1c0–b1c4` 在 batch2 中继续测试（对应 `b2c7/8/9/15/16`）。
# MAGIC    我们只用前 100 个循环做特征，而这部分数据都在 batch1 里 —— 所以只需 **把寿命标签加上续测的循环数**，再删除 batch2 中的这 5 条；
# MAGIC 3. **batch3**：`b3c2/23/32/37/42/43` 噪声大或提前终止 → 剔除；
# MAGIC 4. 兜底：剔除寿命为 NaN、或记录不足 100 个循环的电芯。
# MAGIC
# MAGIC 数据集划分沿用论文：batch1+2 交替分为 train / test，batch3 作为 **secondary test**（不同时间开始的实验，检验泛化能力）。

# COMMAND ----------

DROP = {"b1c8", "b1c10", "b1c12", "b1c13", "b1c22",
        "b3c2", "b3c23", "b3c32", "b3c37", "b3c42", "b3c43"}
CARRY_OVER = {"b1c0": ("b2c7", 662), "b1c1": ("b2c8", 981), "b1c2": ("b2c9", 1060),
              "b1c3": ("b2c15", 208), "b1c4": ("b2c16", 482)}

meta = meta_pdf.copy()
if {1, 2}.issubset(set(meta.batch)):
    for b1, (b2, extra) in CARRY_OVER.items():
        meta.loc[meta.cell_id == b1, "cycle_life"] += extra
    DROP |= {b2 for b2, _ in CARRY_OVER.values()}
else:
    # 只处理了部分批次时，跨批次电芯的寿命不完整，直接剔除
    DROP |= set(CARRY_OVER) | {b2 for b2, _ in CARRY_OVER.values()}

meta = meta[~meta.cell_id.isin(DROP)]
meta = meta[meta.cycle_life.notna() & meta.cell_id.isin(curves_pdf.cell_id)].copy()

# 论文划分：batch1+2 按顺序交替（偶数位 test、奇数位 train，另外第 83 个进 test）；batch3 = secondary_test
meta["_order"] = meta.cell_id.str.extract(r"b(\d)c(\d+)").astype(int).apply(tuple, axis=1)
meta = meta.sort_values("_order").drop(columns="_order").reset_index(drop=True)
is_b12 = meta.batch.isin([1, 2])
pos = pd.Series(np.nan, index=meta.index)
pos[is_b12] = np.arange(is_b12.sum())
meta["split"] = np.where(~is_b12, "secondary_test",
                         np.where((pos % 2 == 0) | (pos == 83), "test", "train"))
meta["cycle_life"] = meta.cycle_life.astype(int)

print(meta.groupby(["batch", "split"]).size())
print("循环寿命范围:", meta.cycle_life.min(), "–", meta.cycle_life.max())

# COMMAND ----------

# MAGIC %md
# MAGIC ## ④ 写入 Delta 表

# COMMAND ----------

keep = set(meta.cell_id)

# 👀 界面：Tables 出现 cell_metadata / cell_cycle_summary / cell_early_curves 三张表
(spark.createDataFrame(meta)
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(CELL_META_TABLE))

cyc = cycles_pdf[cycles_pdf.cell_id.isin(keep)].copy()
cyc["cycle"] = cyc["cycle"].astype(int)
(spark.createDataFrame(cyc)
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(CELL_CYCLE_TABLE))

(spark.createDataFrame(curves_pdf[curves_pdf.cell_id.isin(keep)])
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(CELL_CURVE_TABLE))

spark.sql(f"COMMENT ON TABLE {CELL_META_TABLE} IS 'Severson et al. 2019 电芯元数据（CC BY 4.0）：充电策略、循环寿命、数据集划分'")
spark.sql(f"COMMENT ON TABLE {CELL_CYCLE_TABLE} IS '每电芯每循环汇总：qd 放电容量(Ah)、qc 充电容量、ir 内阻(Ω)、温度(℃)、chargetime 充电时间(min)'")
spark.sql(f"COMMENT ON TABLE {CELL_CURVE_TABLE} IS '第10/100循环按电压(3.6V→2.0V)线性插值的放电容量曲线，各1000点'")
print(f"✅ {len(meta)} 个电芯、{len(cyc)} 行循环数据已写入")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 先看看数据：容量衰减曲线
# MAGIC 这是电池领域最经典的图：每条线是一个电芯，x=循环次数，y=放电容量。注意寿命差异有多大（同一型号电芯，只是快充策略不同）。

# COMMAND ----------

display(spark.sql(f"""
SELECT s.cell_id, s.cycle, s.qd, m.cycle_life, m.charge_policy
FROM {CELL_CYCLE_TABLE} s JOIN {CELL_META_TABLE} m USING (cell_id)
WHERE s.qd BETWEEN 0.8 AND 1.15 AND s.cycle % 10 = 0
ORDER BY cell_id, cycle
"""))
# 👉 在结果区点 “+ → Visualization”，选 Line 图：X=cycle，Y=qd，Group by=cell_id
