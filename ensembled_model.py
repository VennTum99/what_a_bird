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
image_path = "./yolov4/bird_data/detect_image/"
test_dir = './325-bird-species/test/'
result_path = './yolov4/bird_data/result_image/'

# Load Images
test_data =  tf.keras.preprocessing.image_dataset_from_directory(test_dir, label_mode='categorical',
                                                                 image_size=(224, 224), batch_size=32)
test_labels = test_data.class_names

detect_generator = tf.keras.preprocessing.image_dataset_from_directory(image_path, label_mode='categorical',
                                                                 image_size=(224, 224), batch_size=1, shuffle=False, seed = 1)

import numpy as np
from skimage.transform import resize

plt.figure(figsize=(16,16))

n = 3

for i, (img, label) in enumerate(detect_generator.take(len(detect_generator))):
    n_img = len(label)

    eff_img = img
    img = img.numpy() / 255.0

    model_prediction = vitnet.predict(img)
    eff_prediction = effnet.predict(eff_img)
    
    # Ensemble : Weight Averaging
    model_prediction = (model_prediction + eff_prediction) / 2
    
    for j in range(1):
        plt.subplot(2, 2, j + 1)
        plt.imshow((img[j] * 255).astype('uint8'))
        
        top_n = np.sort(model_prediction[j])[-n:]
        top_n_idx = np.argsort(model_prediction[j])[-n:]
        
        # OOD Detection : Based on "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks", 2017 ICLR
        # Labeling Top-3 Probability labels
        if top_n[-1] > 0.9:
            plt.title(f"Predicted Species: {test_labels[tf.argmax(tf.round(model_prediction[j]))]}\n1st Candidate: {test_labels[top_n_idx[-1]]} {top_n[-1] : 0.6f}\n2nd Candidate: {test_labels[top_n_idx[-2]]} {top_n[-2] : 0.6f}\n3rd Candidate: {test_labels[top_n_idx[-3]]} {top_n[-3] : 0.6f}")
        else:
            plt.title(f"Predicted Species: Not in DataSet\n1st Candidate: {test_labels[top_n_idx[-1]]} {top_n[-1] : 0.6f}\n2nd Candidate: {test_labels[top_n_idx[-2]]} {top_n[-2] : 0.6f}\n3rd Candidate: {test_labels[top_n_idx[-3]]} {top_n[-3] : 0.6f}")

        plt.axis("off")
        
        # Save Predicted Images
        plt.savefig(result_path + 'bird_' + str(i) + '.png', dpi = 50)
    
    if len(detect_generator) - 1 <= i: break

