# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:10:39 2021

@author: love2
"""
import os
import shutil
from more_itertools import chunked
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("default")
import keras
from keras.utils.data_utils import get_file
from keras.utils.np_utils import to_categorical
from keras import layers
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split

## why doesn't imagedatagenerator work????

labels = ["Tomato___Bacterial_spot",
"Tomato___Early_blight",
"Tomato___healthy",
"Tomato___Late_blight",
"Tomato___Leaf_Mold",
"Tomato___Septoria_leaf_spot",
"Tomato___Spider_mites Two-spotted_spider_mite",
"Tomato___Target_Spot",
"Tomato___Tomato_mosaic_virus",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus"]


x = []
y = []
test_x = []
test_y = []


for label in labels:
    doi = os.path.join(r"C:\Users\love2\PlantVillage-Dataset\Tomato",label)
    paths = os.listdir(doi)
    paths = [x for x in paths if x.endswith(".JPG")]
    for path in paths:
        img_path = os.path.join(doi,path)
        img = load_img(img_path, grayscale=False, color_mode='rgb', target_size=(32,32), interpolation='nearest')
        img = img_to_array(img)
        img /= 255.
        
        if label == "healthy":
            idx = 0
        elif label == "Tomato___Bacterial_spot":
            idx = 1
        elif label == "Tomato___Early_blight":
            idx = 2
        elif label == "Tomato___Late_blight":
            idx = 3
        elif label == "Tomato___Leaf_Mold":
            idx = 4
        elif label == "Tomato___Septoria_leaf_spot":
            idx = 5
        elif label == "Tomato___Spider_mites Two-spotted_spider_mite":
            idx = 6
        elif label == "Tomato___Target_Spot":
            idx = 7
        elif label == "Tomato___Tomato_mosaic_virus":
            idx = 8
        elif label == "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
            idx = 9
        
        
        x.append(img)
        y.append(idx)


x = np.array(x)
y = np.array(y)


#this function converts numbers to one hot labels. ex. 0 -> [1,0,0] 1->[0,1,0], 2->[0,0,1]
y = to_categorical(y,num_classes=10)


train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=1,stratify = y)

print(train_x.shape,train_y.shape)
print(test_x.shape,test_y.shape)
train_x, valid_x, train_y, valid_y = train_test_split(train_x,train_y,test_size=0.2)
print(train_x.shape,train_y.shape)
print(test_x.shape,test_y.shape)
print(valid_x.shape, valid_y.shape)
plt.subplot(1,2,1)
img = load_img(r"C:\Users\love2\PlantVillage-Dataset\Tomato\Tomato___Bacterial_spot\00a7c269-3476-4d25-b744-44d6353cd921___GCREC_Bact.Sp 5807.jpg",target_size=(64,64))
x = img_to_array(img)/255.
plt.imshow(x)

plt.subplot(1,2,2)
img = load_img(r"C:\Users\love2\PlantVillage-Dataset\Tomato\Tomato___Bacterial_spot\00a7c269-3476-4d25-b744-44d6353cd921___GCREC_Bact.Sp 5807.jpg",target_size=(64,64))
x = img_to_array(img)/255.
plt.imshow(x)
plt.show()
# build model
# Increasing the input size for the CNN and Data Augmentation can further enhance the accuracy.
model = Sequential([
    #feature extraction layer
    
    #block1
    layers.Conv2D(64,(3,3),padding="same",name="block1_conv1",input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Conv2D(64,(3,3),padding="same",name="block1_conv2"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2,2),strides=(2,2),name="block1_pool"),
    #block2
    layers.Conv2D(128,(3,3),padding="same",name="block2_conv1"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Conv2D(128,(3,3),padding="same",name="block2_conv2"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2,2),strides=(2,2),name="block2_pool"),
    #block3
    layers.Conv2D(256,(3,3),padding="same",name="block3_conv1"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Conv2D(256,(3,3),padding="same",name="block3_conv2"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Conv2D(256,(3,3),padding="same",name="block3_conv3"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2,2),strides=(2,2),name="block3_pool"),

    #block4
    layers.Conv2D(512,(3,3),padding="same",name="block4_conv1"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Conv2D(512,(3,3),padding="same",name="block4_conv2"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Conv2D(512,(3,3),padding="same",name="block4_conv3"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2,2),strides=(2,2),name="block4_pool"),

    #block5
    layers.Conv2D(512,(3,3),padding="same",name="block5_conv1"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Conv2D(512,(3,3),padding="same",name="block5_conv2"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Conv2D(512,(3,3),padding="same",name="block5_conv3"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2,2),strides=(2,2),name="block5_pool"),

    layers.Flatten(),
    
    #inference layer
    layers.Dense(512,name="fc1"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.5),
    
    layers.Dense(512,name="fc2"),
    layers.BatchNormalization(),
    layers.Activation("relu"),    
    layers.Dropout(0.5),
    
    layers.Dense(10,name="prepredictions"),
    layers.Activation("softmax",name="predictions")
    
])

model.compile(optimizer = "adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
history = model.fit(train_x,train_y,validation_data=(valid_x,valid_y),batch_size=32, epochs=25)
plt.plot(history.history["accuracy"],label="train_accuracy")
plt.plot(history.history["val_accuracy"],label="validation_accuracy")
plt.legend()
plt.show()

plt.plot(history.history["loss"],label="train_loss")
plt.plot(history.history["val_loss"],label="validation_loss")
plt.legend()
plt.show()
scores = model.evaluate(test_x, test_y)
print(f"Test Accuracy: {scores[1]*100}")
n = 10
input_image = test_x[n][np.newaxis,...]
print("label is: ", test_y[n])

predictions = model.predict(input_image)
print("prediction is",predictions[0])
