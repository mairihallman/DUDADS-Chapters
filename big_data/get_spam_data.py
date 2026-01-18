import urllib.request
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"

selected_features = {
    0: 'make', 1: 'address', 2: 'all', 4: 'our', 5: 'over',
    6: 'remove', 7: 'internet', 8: 'order', 9: 'mail',
    10: 'receive', 11: 'will', 12: 'people', 13: 'report',
    14: 'addresses', 15: 'free', 16: 'business', 17: 'email',
    18: 'you', 19: 'credit', 20: 'your', 21: 'font',
    23: 'money', 28: 'lab', 29: 'labs', 30: 'telnet',
    32: 'data', 35: 'technology', 37: 'parts', 39: 'direct',
    41: 'meeting', 42: 'original', 43: 'project', 45: 'edu',
    46: 'table', 47: 'conference'
}
fids = sorted(selected_features.keys())
feature_names = [selected_features[i] for i in fids]

spark = SparkSession.builder.getOrCreate()

sc = spark.sparkContext

local_file = "/tmp/spambase.csv"
urllib.request.urlretrieve(url, local_file)

df_raw = spark.read.text(local_file)

cols = [f"c{i}" for i in range(57)] + ["label_raw"]

df = (
    df_raw
    .select(F.split(F.col("value"), ",").alias("parts"))
    .select([F.col("parts")[i].cast("double").alias(cols[i]) for i in range(58)])
)

df = df.withColumn("label", F.col("label_raw").cast("int")).drop("label_raw")

df = df.select(
    "label",
    *[F.col(f"c{i}").alias(selected_features[i]) for i in fids]
)

df = df.dropna()

assembler = VectorAssembler(
    inputCols=feature_names,
    outputCol="features"
)

out = assembler.transform(df)
out_compact = out.select("label", *feature_names, "features")

out_compact.write.mode("overwrite").parquet("big_data/datasets/spambase_35.parquet")

# print("Features order:", feature_names)

spark.stop()
