from pyspark.sql import SparkSession
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.linalg import Vectors
import numpy as np

spark = SparkSession.builder.appName("kmeans_penguins_mapreduce").getOrCreate()

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

# standardize only numeric dims; keep sex_bin untouched
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

def kmeans(M, k=3, max_iters=50, tol=1e-6, seed=0):
    np.random.seed(seed)
    centroids = [np.array(c, dtype=float) for _, c in M.takeSample(False, k, seed)]
    prev_move = float("inf")

    for it in range(max_iters):
        broadcast_centroids = M.context.broadcast(centroids)
        mapped = M.map(lambda lf: (nearest_id(np.array(lf[1]), broadcast_centroids.value),
                                  (np.array(lf[1]), 1)))

        reduced = mapped.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
        mapped_new_centroids = reduced.mapValues(lambda sc: sc[0] / sc[1]).collectAsMap()

        k = len(centroids)
        new_centroids = [mapped_new_centroids.get(j, centroids[j]) for j in range(k)]

        move = sum(np.linalg.norm(nc - c) for nc, c in zip(new_centroids, centroids))
        print(f"iter {it}: centroid movement = {move:.6f}")

        centroids = new_centroids
        if move <= tol or abs(prev_move - move) <= 1e-12:
            break
        prev_move = move

    counts = (mapped.map(lambda kv: (kv[0], kv[1][1]))
                    .reduceByKey(lambda a, b: a + b)
                    .collectAsMap())

    return centroids, counts, mapped

k = 3
centroids_std, counts, mapped = kmeans(M, k=k, seed=0)
assignments = M.map(lambda lf: (nearest_id(lf[1], centroids_std), lf[0]))

pair_counts = (assignments
               .map(lambda cl: ((cl[0], cl[1]), 1))
               .reduceByKey(lambda a, b: a + b))

by_cluster = (pair_counts
              .map(lambda kv: (kv[0][0], (kv[0][1], kv[1])))
              .groupByKey()
              .mapValues(lambda xs: {lab: int(n) for lab, n in xs})
              .collect())

names = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}

for cluster_id, counts in sorted(by_cluster, key=lambda x: x[0]):
    total = sum(counts.values())
    purity = (max(counts.values()) / total) if total else 0.0
    pretty = {names.get(l, l): n for l, n in counts.items()}
    print(f"Cluster {int(cluster_id)}: {pretty}  | purity ≈ {purity:.3f}")

spark.stop()
