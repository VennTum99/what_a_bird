import tensorflow as tf
from tensorflow.keras import backend, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GaussianNoise, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

from vit_keras import vit

# Data path
train_path = "./325-bird-species/train"
valid_path = "./325-bird-species/valid"
test_path = "./325-bird-species/test"

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.1,
)
valid_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    color_mode='rgb',
    class_mode='sparse',
    shuffle=True,
)

validation_generator = valid_datagen.flow_from_directory(
    valid_path,
    target_size=(224, 224),
    batch_size=32,
    color_mode='rgb',
    class_mode='sparse')

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    color_mode='rgb',
    class_mode='sparse')

# Vit pretrained with Imagenet Model load
vit_model = vit.vit_l32(
    image_size=224,
    pretrained=True,
    include_top=False,
    pretrained_top=False
)

def scheduler(epoch, lr):
    if epoch != 0 and epoch % 7 == 0:
        return lr * 0.1
    else:
        return lr

lr_scheduler = LearningRateScheduler(scheduler)

finetune_at = 28

# Fine Tuning
for layer in vit_model.layers[:finetune_at - 1]:
    layer.trainable = False

# Add GaussianNoise layer
noise = GaussianNoise(0.01, input_shape=(224, 224, 3))

# Classification head
head = Dense(325, activation="softmax")

# ViT Model Sequence
model = Sequential()
model.add(noise)
model.add(vit_model)
model.add(head)

model.compile(optimizer=optimizers.Adam(),
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])

# Model train by train data with Early Stopping
history = model.fit(
          train_generator,
          epochs=100,
          validation_data=validation_generator,
          verbose=1, 
          shuffle=True,
          callbacks=[
              EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True),
              lr_scheduler,
          ])

history_dict = history.history
val_acc_values = history_dict["val_accuracy"]

print("best val_acc: ", np.max(val_acc_values), "epoch: ", np.argmax(val_acc_values))

# Test Vit Model by test dataset
test_loss, test_acc = model.evaluate(test_generator)

print("test_acc: ", test_acc)

# Save trained Model
model.save('./saved_model/vit_best_temp.h5')
