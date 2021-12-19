import tensorflow as tf
from tensorflow.keras import backend, optimizers
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow import keras 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from tensorflow.keras.layers import GaussianNoise, Dense

import numpy as np

import matplotlib.pyplot as plt

from vit_keras import vit

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession



# load EfficientNet B5 pretrained Model Process
eff_base_model = tf.keras.applications.EfficientNetB5(include_top= False, weights = "imagenet")

for layer in eff_base_model.layers[:-5]:
  eff_base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224,3))

x = eff_base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(325, activation="softmax")(x)

effnet = tf.keras.Model(inputs,outputs)

effnet.compile(
    loss = tf.keras.losses.categorical_crossentropy,
    optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001),
    metrics = ["accuracy"]
)

effnet.load_weights('./saved_model/effnet_best.h5')

# Load ViT pretrained Model Process
vit_base_model = vit.vit_l32(
    image_size=224,
    pretrained=True,
    include_top=False,
    pretrained_top=False
)

finetune_at = 28

# Fine Tuning
for layer in vit_base_model.layers[:finetune_at - 1]:
    layer.trainable = False

noise = GaussianNoise(0.01, input_shape=(224, 224, 3))
head = Dense(325, activation="softmax")

vitnet = Sequential()
vitnet.add(noise)
vitnet.add(vit_base_model)
vitnet.add(head)

vitnet.compile(optimizer=optimizers.Adam(),
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])

vitnet.load_weights('./saved_model/vit_best.h5')



# Main : Predict the detected images by Ensembled Model

# Image Path
test_dir = './325-bird-species/test/'

# Load Images
test_generator =  tf.keras.preprocessing.image_dataset_from_directory(test_dir, label_mode='categorical',
                                                                 image_size=(224, 224), batch_size=32)

test_datagen = ImageDataGenerator(rescale=1/255)
vit_test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    color_mode='rgb',
    class_mode='sparse')

test_labels = test_generator.class_names

print('Vision Transformer Test Accuracy')
test_loss, test_acc = vitnet.evaluate(vit_test_generator)
print('Final ViT Accuracy : ' + str(test_acc))

print('Efficient Net Test Accuracy')
test_loss, test_acc = effnet.evaluate(test_generator)
print('Final Efficient Net Accuracy : ' + str(test_acc))

import numpy as np
from skimage.transform import resize

n = 3

true = 0
total = 0

print('Ensembled Model Test Accuracy')

for i, (img, label) in enumerate(test_generator.take(len(test_generator))):
    n_img = len(label)

    eff_img = img
    img = img.numpy() / 255.0

    model_prediction = vitnet.predict(img)
    eff_prediction = effnet.predict(eff_img)
    
    # Ensemble : Weight Averaging
    model_prediction = (model_prediction + eff_prediction) / 2
    
    for j in range(model_prediction.shape[0]):
        if(tf.argmax(tf.round(model_prediction[j])) == tf.argmax(label[j])): true = true + 1
    
        total = total + 1
    
    print(str(i) + 'th Batch Test Accuracy : ' + str(1.0 * true / total))
    
    if len(test_generator) - 1 <= i: break

