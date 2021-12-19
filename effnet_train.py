
import tensorflow as tf
from tensorflow.keras import backend, optimizers
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import keras

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

# Dataset Load
train_path = "./325-bird-species/train"
valid_path = "./325-bird-species/valid"
test_path = "./325-bird-species/test"

train_data = tf.keras.preprocessing.image_dataset_from_directory(train_path, label_mode='categorical',
                                                                 image_size=(224, 224), batch_size=32)

test_data =  tf.keras.preprocessing.image_dataset_from_directory(test_path, label_mode='categorical',
                                                                 image_size=(224, 224), batch_size=32)

valid_data =  tf.keras.preprocessing.image_dataset_from_directory(valid_path, label_mode='categorical',
                                                                image_size=(224, 224), batch_size=32)

# Data augmentation
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1, 0.1)
    ]
)

# EfficientNetB5 setting
base_model = tf.keras.applications.EfficientNetB5(include_top=False, weights='imagenet')

# Fine Tuning
for layer in base_model.layers[:-5]:
    base_model.trainable = False
    
# EfficientNet Model sequence
inputs = tf.keras.Input(shape=(224, 224,3))

x = data_augmentation(inputs)
x = base_model(x)
x = layers.GlobalAveragePooling2D()(x)

outputs = layers.Dense(325, activation='softmax')(x)

effnet = tf.keras.Model(inputs, outputs)

effnet.compile(loss = tf.keras.losses.categorical_crossentropy,
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = ['accuracy'])

# EfficientNet training with EarlyStopping
effnet.fit(train_data,
               epochs=100,
               validation_data = valid_data,
               callbacks=[
                  EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True)
              ])

test_loss, test_acc = effnet.evaluate(test_data)

print("test_acc: ", test_acc)

# save EfficientNet Model
effnet.save('./saved_model/effnet_best_temp.h5')
