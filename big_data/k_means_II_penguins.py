from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

spark = SparkSession.builder.getOrCreate()

df_path = "big_data/datasets/penguins_5_synth_gc.parquet"

# select features by name so we can standardize
df = spark.read.parquet(df_path).select(
    "species",
    "sex_bin",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
)

numeric = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

# select numeric features
num_assembler = VectorAssembler(
    inputCols=numeric,
    outputCol="cont_vec"
)

# scale numeric features
scaler = StandardScaler(
    inputCol="cont_vec",
    outputCol="cont_z",
    withMean=True,
    withStd=True
)

# add sex
assembler = VectorAssembler(
    inputCols=["cont_z", "sex_bin"],
    outputCol="features_std"
)

k = 3
kmeans = KMeans(
    k=k,
    seed=0,
    maxIter=50,
    featuresCol="features_std",
    predictionCol="prediction",
    initMode="k-means||",
    initSteps=8,
    tol=1e-6
)

pipeline = Pipeline(stages=[num_assembler, scaler, assembler, kmeans])

model = pipeline.fit(df)
predictions = model.transform(df)

counts = (
    predictions
    .groupBy("prediction", "species")
    .count()
)

rows = counts.collect()
by_cluster = {}
for r in rows:
    by_cluster.setdefault(int(r["prediction"]), {})[int(r["species"])] = int(r["count"])

names = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}

for cid in sorted(by_cluster.keys()):
    d = by_cluster[cid]
    total = sum(d.values())
    purity = (max(d.values()) / total) if total else 0.0
    pretty = {names.get(l, l): n for l, n in d.items()}
    print(f"Cluster {cid}: {pretty}  | purity ≈ {purity:.3f}")

spark.stop()
