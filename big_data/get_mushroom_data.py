import pandas as pd
from ucimlrepo import fetch_ucirepo

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

label = "class"

numeric = ["cap-diameter", "stem-height", "stem-width"]

categorical = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "does-bruise-or-bleed",
    "gill-attachment",
    "gill-spacing",
    "gill-color",
    "stem-root",
    "stem-surface",
    "stem-color",
    "veil-type",
    "veil-color",
    "has-ring",
    "ring-type",
    "spore-print-color",
    "habitat",
    "season",
]

# fetch data from UCI
secondary_mushroom = fetch_ucirepo(id=848)

X = secondary_mushroom.data.features
y = secondary_mushroom.data.targets

# pandas dataframe
df_pd = pd.concat([y, X], axis=1)

spark = SparkSession.builder.getOrCreate()

# spark dataframe
df = spark.createDataFrame(df_pd)


# re-code labels
df = df.withColumn(
    "label",
    F.when(F.col("class") == "e", 0.0)  # edible 
     .when(F.col("class") == "p", 1.0) # poisonous
     .otherwise(None)
)

indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=f"{c}_idx",
        handleInvalid="keep"
    )
    for c in categorical
]

encoder = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in categorical],
    outputCols=[f"{c}_oh" for c in categorical],
    handleInvalid="keep"
)

assembler = VectorAssembler(
    inputCols=numeric + [f"{c}_oh" for c in categorical],
    outputCol="features"
)

pipe = Pipeline(stages=indexers + [encoder, assembler])
model = pipe.fit(df)
out = model.transform(df)

out_compact = out.select(
    "label",
    *numeric,
    *categorical,
    "features"
)

out_compact.write.mode("overwrite").parquet("big_data/datasets/mushrooms_secondary.parquet")
print("file saved sucessfully")
spark.stop()
