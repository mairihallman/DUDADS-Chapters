#region CNN example 1
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

input_shape = (32, 32, 3)
inputs = Input(shape=input_shape)

x = layers.Conv2D(32, (3, 3), activation='relu')(inputs) # convolution and detector step
x = layers.MaxPooling2D((2, 2))(x) # pooling step

# flattenting for classification
x = layers.Flatten()(x)
outputs = layers.Dense(10, activation='softmax')(x)

# Create the model
model = models.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
#endregion

