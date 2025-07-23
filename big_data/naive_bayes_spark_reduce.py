from pyspark import SparkContext

# Initialize SparkContext
# This is usually done automatically in Spark shells or notebooks
# For a standalone script, you might do:
# sc = SparkContext("local[*]", "NaiveBayesExample")
# For production, it would connect to a cluster manager like Mesos [7]
sc = SparkContext("local", "NaiveBayesExample")

# 0. Initial Data Loading: Create an RDD from your list of records M_k
# In a real scenario, this would likely be loaded from HDFS or another distributed storage [7-9]
M_k = [
    {'id': 'r1', 'feature_1': 10, 'feature_2': 5, 'feature_n': 12},
    {'id': 'r2', 'feature_1': 8, 'feature_2': 15, 'feature_n': 10},
    {'id': 'rDk', 'feature_1': 14, 'feature_2': 7, 'feature_n': 18}
]

# Convert the list of records into a Spark RDD
data_rdd = sc.parallelize(M_k)

print("--- Initial Data RDD (records) ---")
print(data_rdd.collect())
# Expected output: [{'id': 'r1', ...}, {'id': 'r2', ...}, {'id': 'rDk', ...}]

# 1. The mapper function equivalent (`flatMap`):
# The MapReduce 'map' function takes an input pair and produces a set of intermediate key/value pairs [10].
# In Spark, the `flatMap` transformation is used to take each record `r` and produce multiple `(key, value)` pairs.
# `flatMap` maps each input value to one or more outputs, similar to the Map function in MapReduce [11].
# Here, for each record, we want to create a pair for each feature.

mapped_rdd = data_rdd.flatMap(lambda record: \
    [(key, value) for key, value in record.items() if key.startswith('feature_')])

print("\n--- Mapped RDD (feature, value pairs) ---")
print(mapped_rdd.collect())
# Expected output: [('feature_1', 10), ('feature_2', 5), ('feature_n', 12),
#                   ('feature_1', 8), ('feature_2', 15), ('feature_n', 10),
#                   ('feature_1', 14), ('feature_2', 7), ('feature_n', 18)]

# 2. All values with the same key are grouped and then reduced (`reduceByKey`):
# In MapReduce, after the map phase, the library groups together all intermediate values associated with the same intermediate key [10]. Then, the reduce function merges these values [12].
# Spark's `reduceByKey` transformation combines these two steps (grouping and reducing) efficiently [11, 13].
# It automatically groups elements by key and then applies an aggregation function (like `add` or summation in your example) to all values associated with each unique key.
# This is similar to how a MapReduce `Combiner` function can do partial merging before network transfer, saving bandwidth [14, 15].

# The `add` function is represented by the `+` operator in Python, or a lambda function.
reduced_rdd = mapped_rdd.reduceByKey(lambda a, b: a + b) # or simply mapped_rdd.reduceByKey(add) if 'add' is defined or imported

print("\n--- Reduced RDD (sum of features) ---")
print(reduced_rdd.collect())
# Expected output: [('feature_1', 32), ('feature_2', 27), ('feature_n', 40)]
# (10+8+14 = 32, 5+15+7 = 27, 12+10+18 = 40)

# Stop the SparkContext
sc.stop()