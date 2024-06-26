import tensorflow as tf

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', # how improvement is quantified'
    min_delta=0, # decrease in monitored metric required to be considered an "improvement"
    patience=3, # number of epochs without improvement after which the model will stop training
    restore_best_weights=True, # whether to restore weights from the best epoch (True) or last epoch (False); should be set to True
    start_from_epoch=0 # how many epochs to wait before checking for early stop
)

model = keras.models.Sequential([
    # define your model here
        ]
)

model.compile(
    # ...
)
history = model.fit(
    # ...
    callbacks=[callback]
)