import tensorflow as tf

n = 10 # desired number of units in layer

# dense layer with L2 weight decay
l2_regularizer=tf.keras.regularizers.L2(
    l2=0.01 # hyperparameter for l2 regularization, default is 0.01
)

layer = tf.keras.layers.Dense(n, regularizer=l2_regularizer)

# dense layer with L1 weight decay
l1_regularizer=tf.keras.regularizers.L1(
    l1=0.01 # hyperparameter for l2 regularization, default is 0.01
)

layer = tf.keras.layers.Dense(n, regularizer=l1_regularizer)

# dense layer with L2 and L1 weight decay
l1l2_regularizer=tf.keras.regularizers.L1L2(
    l1=0.01, # hyperparameter for l2 regularization, default is 0.01
    l2=0.01 # hyperparameter for l2 regularization, default is 0.01
)

layer = tf.keras.layers.Dense(n, regularizer=l1l2_regularizer)

