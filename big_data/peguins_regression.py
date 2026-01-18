from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("penguins_bill_length").getOrCreate()

path = "big_data/datasets/penguins_5_synth_gc.parquet"
df = spark.read.parquet(path)

label = "bill_length_mm"
numeric = ["bill_depth_mm", "flipper_length_mm", "body_mass_g"]
binary = ["sex_bin"]

df = df.withColumnRenamed(label, "label")


num_assembler = VectorAssembler(
    inputCols=numeric,
    outputCol="num_vec"
)

scaler = StandardScaler(
    inputCol="num_vec",
    outputCol="num_z",
    withMean=True,
    withStd=True
)

final_assembler = VectorAssembler(
    inputCols=["num_z"] + binary,
    outputCol="features_lr"
)

# lasso
lr = LinearRegression(
    featuresCol="features_lr",
    regParam = 0.1,
    elasticNetParam= 1,
    labelCol="label",
    maxIter=100
)

# pipeline - regularization plus regression
pipeline = Pipeline(stages=[
    num_assembler,
    scaler,
    final_assembler,
    lr
])

train, test = df.randomSplit([0.8, 0.2], seed=0)

model = pipeline.fit(train)
pred = model.transform(test)
model_info = model.stages[-1]

print("Intercept:", model_info.intercept)
print("Coefficients:", model_info.coefficients)

rmse = RegressionEvaluator(metricName="rmse").evaluate(pred)
r2 = RegressionEvaluator(metricName="r2").evaluate(pred)

print("\nTest RMSE:", rmse)
print("Test R2:", r2)

spark.stop()
