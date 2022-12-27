import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import requests
import seaborn as sns
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import tensorflow as tf
import cv2
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras import datasets, layers, models
import tensorflow as tf

import os
import shutil
testdir  = 'Test_Scripts\\SignatureTestSamples\\'
#read all files in the directory
files = os.listdir(testdir)
try:
    os.mkdir('personA')
    os.mkdir('personB')
    os.mkdir('personC')
    os.mkdir('personD')
    os.mkdir('personE')
    
except: pass

for file in files:
    #move file to the person directory
    shutil.copy(testdir+file, file.split('_')[0])

test = tf.keras.preprocessing.image_dataset_from_directory(
    "Test_Scripts\\Test",
    labels="inferred",
    label_mode="categorical",  # categorical, binary
    # class_names=['0', '1', '2', '3', ...]
    color_mode="rgb",
    batch_size=32,
    image_size=(215, 90),  # reshape if not in this size
    shuffle=True,
    seed=123,
    )

model = tf.keras.models.load_model('D:\\Nourhan\\handwritten-signature-recognition\\model\\resnet_class_sig.h5')
model.evaluate(test)