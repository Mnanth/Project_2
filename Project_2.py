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
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#Step 1: Building A NN Structure
#input shape
input_shape = (500,500,3)

#Relative Paths
BASE_DIR = Path(__file__).resolve().parent
train_path = BASE_DIR / 'data' / 'train'
validation_path = BASE_DIR / 'data' / 'validation'
test_path = BASE_DIR / 'data' / 'test'

#Data augmentation
train_datagen = ImageDataGenerator(
    rescale = 1.0/255,
    shear_range = 0.1,
    zoom_range = 0.1)

val_datagen = ImageDataGenerator(
    rescale = 1.0/255)

