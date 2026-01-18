import pandas as pd
from ucimlrepo import fetch_ucirepo
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import train_test_split

target = "hospital_outcome_1alive_0dead"
continuous_preds = ["age_years","episode_number"]
bin_preds = ["sex_0male_1female"]

sepsis = fetch_ucirepo(id=827)

X = sepsis.data.features
y = sepsis.data.targets

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3, # more original data this time (SMOTE will give us more training data anyways)
    random_state=0,
    stratify=y
)

# standardize training data and apply the same transformation to the test data (avoid leakage)
mean = X_train[continuous_preds].mean()
std  = X_train[continuous_preds].std(ddof=0) # population-level

X_train[continuous_preds] = (X_train[continuous_preds] - mean) / std
X_test[continuous_preds]  = (X_test[continuous_preds]  - mean) / std

# SMOTE on training data ONLY
sm = BorderlineSMOTE(k_neighbors=2, m_neighbors=5,random_state=0)
X_train,y_train = sm.fit_resample(X_train,y_train)

df_train_r = pd.concat([y_train, X_train], axis=1)
df_test_r = pd.concat([y_test, X_test], axis=1)

spark = SparkSession.builder.appName("sepsis_parquet").getOrCreate()

df_train = spark.createDataFrame(df_train_r)
df_test = spark.createDataFrame(df_test_r)

keep_cols = [target] + continuous_preds + bin_preds
df_train = df_train.select(*keep_cols)
df_test = df_test.select(*keep_cols)

df_train = df_train.dropna(subset=keep_cols)
df_test = df_test.dropna(subset=keep_cols)

num_assembler = VectorAssembler(
    inputCols=continuous_preds,
    outputCol="cont_vec"
)

assembler = VectorAssembler(
    inputCols=["cont_vec"] + bin_preds,
    outputCol="features"
)

pipe = Pipeline(stages=[num_assembler, assembler])
pipe_model = pipe.fit(df_train)

train_out = pipe_model.transform(df_train).select(
    F.col(target).alias("label"),
    "features"
)

test_out = pipe_model.transform(df_test).select(
    F.col(target).alias("label"),
    "features"
)

train_out.write.mode("overwrite").parquet("big_data/datasets/sepsis_train.parquet")
test_out.write.mode("overwrite").parquet("big_data/datasets/sepsis_test.parquet")

spark.stop()