import numpy as np
import pandas as pd
from palmerpenguins import load_penguins
from synthpop import MissingDataHandler, DataProcessor, GaussianCopulaMethod

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from pyspark.ml.linalg import VectorUDT


n_samples = 100_000


num_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

species_to_idx = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
idx_to_species = {v: k for k, v in species_to_idx.items()}

spark = (
    SparkSession.builder
    .appName("MakePenguins_UnifiedParquet_GC_3Models_GlobalZ")
    .master("local[*]")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.driver.host", "127.0.0.1")
    .getOrCreate()
)

df = load_penguins()
df = df[["species", "sex"] + num_cols].copy()

df["sex"] = df["sex"].astype("string").str.strip().str.lower()
df = df[df["sex"].isin(["male", "female"])]

df = df.dropna(subset=["species"] + num_cols).reset_index(drop=True)

df["species"] = df["species"].map(species_to_idx).astype(int)
df["sex"] = df["sex"].astype("category")

counts = df["species"].value_counts().sort_index()
probs = (counts / counts.sum()).to_numpy()

expected = probs * n_samples
base = np.floor(expected).astype(int)
remainder = n_samples - base.sum()

frac = expected - base
order = np.argsort(-frac)
for k in range(remainder):
    base[order[k]] += 1

samples_per_class = base.tolist()

print("Original class counts:", counts.to_dict())
print("Original class probs :", probs)
print("Synthetic per-class N:", samples_per_class, "sum =", sum(samples_per_class))

dfs_syn = []

fit_means_by_species = {}
fit_stds_by_species = {}

for sp in [0, 1, 2]:
    n_sp = samples_per_class[sp]
    df_sp = df[df["species"] == sp].copy().reset_index(drop=True)

    X = df_sp[num_cols].to_numpy(dtype=float)
    mu = X.mean(axis=0)
    sd = np.maximum(X.std(axis=0), 1e-12)

    fit_means_by_species[sp] = mu
    fit_stds_by_species[sp] = sd

    for j, c in enumerate(num_cols):
        df_sp[c + "_fitz"] = (df_sp[c].astype(float) - mu[j]) / sd[j]

    df_fit = df_sp[["sex"] + [c + "_fitz" for c in num_cols]].copy()

    metadata_sp = MissingDataHandler().get_column_dtypes(df_fit)
    processor_sp = DataProcessor(metadata_sp)
    df_fit_proc = processor_sp.preprocess(df_fit)

    if df_fit_proc.isna().any().any():
        before = len(df_fit_proc)
        df_fit_proc = df_fit_proc.dropna(axis=0, how="any").reset_index(drop=True)
        after = len(df_fit_proc)
        print(f"[species={sp}] Dropped {before - after} rows with NaNs.")

    gc_sp = GaussianCopulaMethod(metadata_sp)
    gc_sp.fit(df_fit_proc)

    syn_sp = gc_sp.sample(n_sp)

    for j, c in enumerate(num_cols):
        zc = c + "_fitz"
        syn_sp[zc] = pd.to_numeric(syn_sp[zc], errors="coerce")
        syn_sp[c] = syn_sp[zc] * sd[j] + mu[j]

    syn_sp["species"] = sp
    dfs_syn.append(syn_sp)

df_synth = pd.concat(dfs_syn, ignore_index=True)

df_synth["species"] = np.rint(pd.to_numeric(df_synth["species"], errors="coerce")).astype(int)

for c in num_cols:
    df_synth[c] = pd.to_numeric(df_synth[c], errors="coerce")

sex_str = df_synth["sex"].astype("string").str.strip().str.lower()
if sex_str.isin(["male", "female"]).any():
    df_synth["sex_bin"] = (sex_str == "male").astype(int)
else:
    df_synth["sex"] = pd.to_numeric(df_synth["sex"], errors="coerce")
    df_synth["sex_bin"] = (df_synth["sex"] > 0.5).astype(int)

keep_cols = ["species"] + num_cols + ["sex_bin"]
df_synth = df_synth.dropna(subset=keep_cols).reset_index(drop=True)

def make_features_row(row):
    feats = [float(row[c]) for c in num_cols] + [float(row["sex_bin"])]
    return Vectors.dense(feats)

rows = []
for _, r in df_synth.iterrows():
    rows.append((
        int(r["species"]),
        float(r["sex_bin"]),
        float(r["bill_length_mm"]),
        float(r["bill_depth_mm"]),
        float(r["flipper_length_mm"]),
        float(r["body_mass_g"]),
        make_features_row(r),
    ))

schema = StructType([
    StructField("species", IntegerType(), nullable=False),
    StructField("sex_bin", DoubleType(), nullable=False),
    StructField("bill_length_mm", DoubleType(), nullable=False),
    StructField("bill_depth_mm", DoubleType(), nullable=False),
    StructField("flipper_length_mm", DoubleType(), nullable=False),
    StructField("body_mass_g", DoubleType(), nullable=False),
    StructField("features", VectorUDT(), nullable=False),
])

spark_df = spark.createDataFrame(rows, schema=schema)
spark_df.write.mode("overwrite").parquet("big_data/datasets/penguins_5_synth_gc.parquet")

spark.stop()


