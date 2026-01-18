from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

spark = SparkSession.builder.getOrCreate()

df_path = "big_data/datasets/mushrooms_secondary.parquet"
df = spark.read.parquet(df_path).select("label", "features")

train, test = df.randomSplit([0.8, 0.2], seed=0)

dt = DecisionTreeClassifier(
    labelCol="label",
    featuresCol="features",
    seed=0
)

pipeline = Pipeline(stages=[dt])
model = pipeline.fit(train)
predictions = model.transform(test)

evaluator_auc = BinaryClassificationEvaluator(metricName="areaUnderROC")
auc = evaluator_auc.evaluate(predictions)
print(f"\nAUC-ROC = {auc:.6f}")

evaluator = MulticlassClassificationEvaluator() 
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
print(f'F1: {f1:.6f}')

accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
print(f'Accuracy: {accuracy:.6f}')

conf = {
    (int(r.label), int(r.prediction)): r["count"]
    for r in predictions.groupBy("label", "prediction").count().collect()
}

print("\nConfusion matrix (rows=true, cols=pred):")
for t in [0, 1]:
    print(f"{t}: {[conf.get((t, p), 0) for p in [0, 1]]}")
spark.stop()
