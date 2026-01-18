from pyspark.sql import SparkSession
from math import log
import numpy as np
from pyspark.mllib.classification import NaiveBayesModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.evaluation import MulticlassMetrics

spark = SparkSession.builder.appName("naive_bayes_filtered").getOrCreate()
sc = spark.sparkContext

parquet_path = "big_data/datasets/spambase_35.parquet"  
df = spark.read.parquet(parquet_path)

# need this later to see which words are associated with spam
feature_word = [c for c in df.columns if c not in ("label", "features")]

M = df.rdd.map(lambda r: (int(r["label"]), r["features"].toArray()))

def get_feature_counts(label_and_feats):
    label, feats = label_and_feats
    return [((label, j), val) for j, val in enumerate(feats)]

mapped_M = M.flatMap(get_feature_counts)

reduced_M = mapped_M.reduceByKey(lambda a, b: a + b)

label_counts = M.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b)

feature_totals = dict(reduced_M.collect())
class_counts   = dict(label_counts.collect())

n = len(M.first()[1])
spam_ratios = []
for j in range(n):
    spam_freq = feature_totals.get((1, j), 0.0) / class_counts[1]
    ham_freq  = feature_totals.get((0, j), 0.0) / class_counts[0]
    if ham_freq > 0:
        spam_ratios.append((spam_freq / ham_freq, j, spam_freq, ham_freq))

top5 = sorted(spam_ratios, reverse=True)[:5]
print("Top 5 spam-associated words:")
for ratio, j, sf, hf in top5:
    print(f"{feature_word[j]}: {ratio:.3f}x ({sf:.3f} vs {hf:.3f})")

print(f"Class counts: {class_counts[0]} ham, {class_counts[1]} spam")

# split into train and test
train_M, test_M = M.randomSplit([0.8, 0.2], seed=0)

# map / reduce on training data
mapped_train = train_M.flatMap(get_feature_counts)
reduced_train = mapped_train.reduceByKey(lambda a, b: a + b)

label_counts_train = train_M.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b)

class_token_mass = (
    reduced_train
    .map(lambda kv: (kv[0][0], kv[1]))   # drop feature index
    .reduceByKey(lambda a, b: a + b)
)

feature_totals_train = dict(reduced_train.collect())
class_counts_train   = dict(label_counts_train.collect())
class_total_mass     = dict(class_token_mass.collect())

classes = sorted(class_counts_train.keys())
N = sum(class_counts_train.values())
pi_arr = np.array([log(class_counts_train[c] / N) for c in classes], dtype=float)

alpha = 0.1
theta_rows = []
for c in classes:
    denom = class_total_mass.get(c, 0.0) + alpha * n
    row = []
    for j in range(n):
        num = feature_totals_train.get((c, j), 0.0) + alpha
        row.append(log(num / denom))
    theta_rows.append(row)
theta_arr = np.array(theta_rows, dtype=float)

# build the model
model = NaiveBayesModel(labels=np.array(classes), pi=pi_arr, theta=theta_arr)

# predictions on test set
preds_and_labels = test_M.map(
    lambda lp: (float(model.predict(Vectors.dense(lp[1]))), float(lp[0]))
)

metrics = MulticlassMetrics(preds_and_labels)

acc = metrics.accuracy
conf = metrics.confusionMatrix().toArray()

correct = int(conf.trace())
test_n  = int(conf.sum())

print(f"\nTest accuracy: {acc:.3f}  [{correct}/{test_n}]")
print("Confusion matrix (rows=true, cols=pred):")
for i, t in enumerate(classes):
    print(f"{t}: {[int(x) for x in conf[i]]}")

sc.stop()