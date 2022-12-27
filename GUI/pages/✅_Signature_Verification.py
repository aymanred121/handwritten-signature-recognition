import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import requests
import seaborn as sns
from streamlit_lottie import st_lottie
from PIL import Image
import cv2
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
import os

st.set_page_config(page_title='Signature Verification', page_icon=':star:')
#f = open("D:\\Nourhan\\handwritten-signature-recognition\\model\\model.json", "r")
#encoder = models.model_from_json(f.read())
encoder = tf.keras.models.load_model('D:\\Nourhan\\handwritten-signature-recognition\\model\\siames123.h5')
#encoder.load_weights('D:\\Nourhan\\handwritten-signature-recognition\\model\\siamese.h5')

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
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


def findSimilar(img1,img2,thres=1):
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

st.markdown("# Signature Verification :white_check_mark:")

animation = load_lottie('https://assets8.lottiefiles.com/packages/lf20_gckvznnm.json')
with st.container():
    left_col, right_col = st.columns(2)
    with left_col:
        ## Input Fields
        st.write("##")
        st.write("##")
        img1 = None
        img2 = None

           
        uploaded_file = st.file_uploader("Upload an Image 1", type=["jpg", "png", "jpeg","tif"])
        if uploaded_file is not None:
            img1 = Image.open(uploaded_file)
            img1.save("ver_test1.png")
            img1 = img1.resize((215,90))
            #st.write("write...")
            
            #im = tf.keras.utils.load_img('ver_test1.png',color_mode='rgb',target_size=(215,90))
            
            #input_arr1 = tf.keras.utils.img_to_array(im)
            #input_arr1 = np.array([input_arr1]) 
            st.image(img1)

        uploaded_file1 = st.file_uploader("Upload an Image 2", type=["jpg", "png", "jpeg","tif"])
        if uploaded_file1 is not None:
            img2 = Image.open(uploaded_file1)
            img2.save("ver_test2.png")
            img2 = img2.resize((215,90))
            #st.write("write...")
            
            #im1 = tf.keras.utils.load_img('ver_test2.png',color_mode='rgb',target_size=(215,90))
            
            #input_arr2 = tf.keras.utils.img_to_array(im1)
            #input_arr2 = np.array([input_arr2]) 
            st.image(img2)
        
        

        if st.button("Detect"):
            label = findSimilar("ver_test2.png","ver_test1.png")
            st.write("Label : ",label[0])
    with right_col:
        st_lottie(animation, height=400, width=350, key='Signature Verification')

