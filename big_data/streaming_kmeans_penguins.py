from pyspark.sql import SparkSession
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.linalg import Vectors
import numpy as np

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

df_path = "big_data/datasets/penguins_5_synth_gc.parquet"
df = spark.read.parquet(df_path).select("species", "features")

M_r = df.rdd.map(lambda r: (int(r["species"]), r["features"].toArray()))

# standardize
d_cont = 4 # first 4 columns are continuous

X_cont = M_r.map(lambda lf: Vectors.dense(lf[1][:d_cont]))
model = StandardScaler(withMean=True, withStd=True).fit(X_cont)

mu = np.array(model.mean)
sd = np.array(model.std)
bc_mu, bc_sd = sc.broadcast(mu), sc.broadcast(sd)

# standardize numeric features
M = M_r.map(lambda lf: (
    lf[0],
    np.r_[ (np.asarray(lf[1], dtype=float)[:d_cont] - bc_mu.value) / bc_sd.value,
           np.asarray(lf[1], dtype=float)[d_cont:] ]
))

def nearest_id(x, centroids):
    best_j, best_d = None, float("inf")
    xv = Vectors.dense(x)
    for j, c in enumerate(centroids):
        d = Vectors.squared_distance(xv, Vectors.dense(c))
        if d < best_d:
            best_d, best_j = d, j
    return best_j

def get_batches(M, num_batches):
    M_indexed = M.zipWithIndex().cache()
    return [
        M_indexed.filter(lambda kv: kv[1] % num_batches == i).map(lambda kv: kv[0]).cache()
        for i in range(num_batches)
    ]

batches = get_batches(M, num_batches=5) # 5 batches

def streaming_kmeans(batches, k=3, alpha=1, seed=0):
    np.random.seed(seed)

    init_batch = None
    for b in batches:
        if not b.isEmpty():
            init_batch = b
            break
    if init_batch is None:
        return [], {}, None

    centroids = [np.array(c, dtype=float) for _, c in init_batch.takeSample(False, k, seed)]
    n = [0.0 for _ in range(k)]
    mapped_last = None

    for t, batch in enumerate(batches):
        if batch.isEmpty():
            print(f"batch {t}: (empty)")
            continue

        broadcast_centroids = batch.context.broadcast(centroids)

        mapped = batch.map(
            lambda lf: (nearest_id(np.array(lf[1]), broadcast_centroids.value),
                        (np.array(lf[1]), 1))
        )
        mapped_last = mapped

        reduced = mapped.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
        stats = reduced.collectAsMap()

        for j in range(k):
            n_old = alpha * n[j]
            if j in stats:
                s_j, m_j = stats[j]
                x_j = s_j / m_j
                denom = n_old + float(m_j)
                centroids[j] = (n_old * centroids[j] + float(m_j) * x_j) / denom
                n[j] = denom
            else:
                n[j] = n_old

        counts_batch = {j: int(stats[j][1]) for j in stats}
        counts_pretty = {j: counts_batch.get(j, 0) for j in range(k)}
        print(f"batch {t}: actual points = {counts_pretty}")

    final_counts = (
        mapped_last.map(lambda kv: (kv[0], kv[1][1]))
        .reduceByKey(lambda a, b: a + b)
        .collectAsMap()
    ) if mapped_last is not None else {}

    return centroids, final_counts, mapped_last

k = 3
centroids_stream, counts_stream, mapped_last = streaming_kmeans(batches, k=k)

assignments = M.map(lambda lf: (nearest_id(lf[1], centroids_stream), lf[0]))

print("\nCluster sizes:", dict(assignments.countByKey()))

pair_counts = (
    assignments
    .map(lambda cl: ((cl[0], cl[1]), 1))
    .reduceByKey(lambda a, b: a + b)
)

by_cluster = (
    pair_counts
    .map(lambda kv: (kv[0][0], (kv[0][1], kv[1])))
    .groupByKey()
    .mapValues(lambda xs: {lab: int(n) for lab, n in xs})
    .collect()
)

names = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}

for cluster_id, counts in sorted(by_cluster, key=lambda x: x[0]):
    total = sum(counts.values())
    purity = (max(counts.values()) / total) if total else 0.0
    pretty = {names.get(l, l): n for l, n in counts.items()}
    print(f"Cluster {int(cluster_id)}: {pretty}  | purity = {purity:.3f}")

spark.stop()
