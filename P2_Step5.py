#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 18:18:23 2024

@author: mithu
"""
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class_labels = ['crack', 'missing-head', 'paint-off']

model =  tf.keras.models.load_model('/Users/mithu/Documents/GitHub/Project_2/model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

BASE_DIR = Path(__file__).resolve().parent
img_path = BASE_DIR / 'Data' / 'test' / 'paint-off'/'test_paintoff.jpg'


def load_and_preprocess_image(img_path, target_size=(500,500)):
    
    img = image.load_img(img_path, target_size=target_size)
    
    img_array = image.img_to_array(img)
    
    img_array = img_array/255
    
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image_class(model, img_path):
    
    processed_image = load_and_preprocess_image(img_path)
    
    predictions = model.predict(processed_image)
    
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    
    print(f"Predicted Class: {class_labels[predicted_class]} with Confidence: {confidence:.4f}")
   
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {class_labels[predicted_class]} ({confidence:.4f})")
    plt.show()
    
predict_image_class(model, img_path)
