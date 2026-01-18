from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("nn_sepsis").getOrCreate()

train_path = "big_data/datasets/sepsis_train.parquet"
train = spark.read.parquet(train_path).select("label", "features")

test_path = "big_data/datasets/sepsis_test.parquet"
test = spark.read.parquet(test_path).select("label", "features")

mlp = MultilayerPerceptronClassifier(layers=[3, 8, 2], seed=0,maxIter=500, stepSize=0.01)

model = mlp.fit(train)

predictions = model.transform(test)

evaluator_auc = BinaryClassificationEvaluator(metricName="areaUnderROC")
auc = evaluator_auc.evaluate(predictions)
print(f"\nAUC-ROC = {auc:.6f}")

evaluator = MulticlassClassificationEvaluator() 
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
print(f'F1: {f1:.6f}')

accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy", evaluator.metricLabel:1})
print(f'Accuracy: {accuracy:.6f}')


conf = {
    (int(r.label), int(r.prediction)): r["count"]
    for r in predictions.groupBy("label", "prediction").count().collect()
}

print("\nConfusion matrix (rows=true, cols=pred):")
for t in [0, 1]:
    print(f"{t}: {[conf.get((t, p), 0) for p in [0, 1]]}")

spark.stop()