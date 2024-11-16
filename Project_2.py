#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:18:29 2024

@author: mithu
"""

import tensorflow as tf
import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt
from pathlib import Path

from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
#Step 1: Building A NN Structure
#input shape
input_shape = (500,500,3)

#Relative Paths
BASE_DIR = Path(__file__).resolve().parent
train_path = BASE_DIR / 'Data' / 'train'
validation_path = BASE_DIR / 'Data' / 'valid'
test_path = BASE_DIR / 'Data' / 'test'

#Data augmentation
train_datagen = ImageDataGenerator(
    rescale = 1.0/255,
    shear_range = 0.1, 
    zoom_range = 0.1)

val_datagen = ImageDataGenerator(
    rescale = 1.0/255)

#Creating the train and validation generator
train_data = image_dataset_from_directory(
    train_path,
    labels = 'inferred',
    label_mode = 'categorical',
    image_size = (500,500),
    batch_size = 32)

validation_data = image_dataset_from_directory(
    validation_path,
    labels = 'inferred',
    label_mode = 'categorical',
    image_size = (500,500),
    batch_size = 32)

test_data = image_dataset_from_directory(
    test_path,
    labels = 'inferred',
    label_mode = 'categorical',
    image_size = (500,500),
    batch_size = 32)


#Step 2.Step 3: Neural Network Architecture Design/ Hyperparameter Analysis
model = Sequential()

model.add(Conv2D (32, (3,3), activation = 'relu', input_shape=(500,500,3)))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add (Dense(128, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(3, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()

history = model.fit(train_data, validation_data=validation_data, epochs = 10)

test_loss,test_acc = model.evaluate(test_data)

print(f'Test Accuracy: {test_acc:.4f}')
#plots
plt.figure(figsize=(12,4))
#Acc
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label = 'Training Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
#Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label = 'Training Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

