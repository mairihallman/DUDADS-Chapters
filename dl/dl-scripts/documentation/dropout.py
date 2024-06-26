import tensorflow as tf

my_dropout = tf.keras.layers.Dropout(
    rate=0.1 # the proportion of units to be randomly dropped out at each iteration
)

model = keras.models.Sequential([
    # ...
    my_dropout,
    # ...
        ]
)