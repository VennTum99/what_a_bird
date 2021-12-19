# What a Bird! : Bird Species Classifier

Welcome to What a Bird!

What a Bird! is a bird species classification model.

It was trained based on kaggle's '325 bird species' dataset.

It can be used in the following steps:

1. Clone the repository.

2. Download the google drive link below.

https://drive.google.com/file/d/1XXXVPvZhugx1eBCne4xpjsjE0qBOI8n6/view?usp=sharing

3. Unzip the downloaded data to the 'what a bird' folder to make the final folier state like 'how_to_unzip.png'! (No need to delete the file, just unzip it so that '325-bird-species' and the rest of the folders are all inside what a bird.)

Now you can run each of them using the notebook in what a bird.

Please install requirements.txt first. (pip install -r requirements.txt)
And you can run each notebook separately.

**However, before running the notebook, please correct the path of the cd in the notebook so that it works properly. Instructions for using each notebook exist within the notebook.**

Discription about Notebooks

## 1. what_a_bird

It is a notebook running what_a_bird that we have completed.

First, detect the bird image using the yolov4/detect.py file. When using, put the image you want to detect as described in './yolov4/bird_data/raw_image/'.

Objects detected through this exist in './yolov4/bird_data/detect_image/bird/'.
(the whole image with bound box is on './yolov4/bird_data/result_image/result.png')
Classify these images through ensembled_model.py.

The image after this classification is located in './yolov4/bird_data/result_image/' along with the label and top-3 candidate captioning. You can download and check the actual image.

In the case of the !python model_accuracy_test.py code located at the end, it is a code to check the test accuracy of the pretrained model in the '/saved_model/' folder where we saved it.

**Solution for undetected bird object**

Image classification can be performed without object detection. Put the image to be classified in './yolov4/bird_data/detect_image/bird/'. Then run ensembled_model.py and the result will be saved in result_image folder.

## 2. data_checker

This is the code to check the structure of the data in the kaggle dataset. Augmented data and image / classes chart can be checked.

## 3. model_train

It is a notebook that stores the trained model in '/saved_model' by learning vit and efficientNet.

Various parameters can be modified by modifying the corresponding python code for each.

The saved model has '_temp' appended to the end so that the existing pretrained model is not overwritten. If you want to use a model that you have created yourself, edit the name as described in the notebook and use it.

## Python Codes

Here's the important code I wrote and modified.

requirements.txt

vit_train.py

effnet_train.py

ensembled_model.py

model_accuracy_test.py

data_checker.ipynb

model_train.ipynb

what_a_bird.ipynb

yolov4/detect.py

yolov4/core/utils.py

Other than that, the rest of the code has been modified appropriately so that the actual model can work.

## Appendix

Test dataset - kaggle 325-bird-species

Yolov4 - MiT_tensorflow-yolov4-tflite

OOD Detection - A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks
