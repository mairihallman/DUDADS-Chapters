from pyspark.sql import SparkSession, functions as F
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

sc = SparkSession.builder.appName("naive_bayes_filtered_df").getOrCreate()

df = sc.read.parquet("big_data/datasets/spambase_35.parquet").select(
    F.col("label").cast("double").alias("label"),
    "features"
)

train_df, test_df = df.randomSplit([0.8, 0.2], seed=0)

model = NaiveBayes(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    smoothing=0.1,
    modelType="multinomial"
).fit(train_df)

pred = model.transform(test_df).select("label", "prediction")

acc = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
).evaluate(pred)

conf = {
    (int(r.label), int(r.prediction)): r["count"]
    for r in pred.groupBy("label", "prediction").count().collect()
}

total = pred.count()
correct = conf.get((0, 0), 0) + conf.get((1, 1), 0)

print(f"Test accuracy: {acc:.3f}  [{correct}/{total}]")
print("Confusion matrix (rows=true, cols=pred):")
for t in [0, 1]:
    print(f"{t}: {[conf.get((t, p), 0) for p in [0, 1]]}")

sc.stop()
