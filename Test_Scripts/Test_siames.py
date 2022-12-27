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
testdir  = 'SignatureTestSamples/'
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
    "Test Scripts/SignatureTestSamples/",
    labels="inferred",
    label_mode="categorical",  # categorical, binary
    # class_names=['0', '1', '2', '3', ...]
    color_mode="rgb",
    batch_size=32,
    image_size=(215, 90),  # reshape if not in this size
    shuffle=True,
    seed=123,
    )

encoder = tf.keras.models.load_model('D:\\Nourhan\\handwritten-signature-recognition\\model\\siames123.h5')
############################################################################################################
def read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128)) 
    return image

def classify_images(face_list1, face_list2, threshold=1.3):
    # Getting the encodings for the passed faces
    tensor1 = encoder.predict(face_list1)
    tensor2 = encoder.predict(face_list2)
    
    distance = np.sum(np.square(tensor1-tensor2), axis=-1)
    prediction = np.where(distance<=threshold, 0, 1)
    return prediction


def findSimilar(img1,img2,thres=1.1):
    a=read_image(img1)
    a=preprocess_input(a)
    lst1=[a]
    lst1=np.array(lst1)
    b=read_image(img2)
    b=preprocess_input(b)
    lst2=[b]
    lst2=np.array(lst2)
    xx=[]
    x=classify_images(lst1,lst2,thres)
    if x ==0: xx.append('real')
    else:xx.append('forged')
    return xx
############################################################################################################

