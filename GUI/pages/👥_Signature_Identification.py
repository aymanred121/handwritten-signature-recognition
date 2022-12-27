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
from tensorflow.keras import datasets, layers, models
import tensorflow as tf

st.set_page_config(page_title='Signature Identification', page_icon=':star:')

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.markdown("# Signature Identification :busts_in_silhouette:")

m = models.load_model('D://Nourhan//handwritten-signature-recognition//model//resnet_class_sig.h5')


result = 0 

animation = load_lottie('https://assets8.lottiefiles.com/packages/lf20_pzk9h5cf.json')
with st.container():
    left_col, right_col = st.columns(2)
    with left_col:
        ## Input Fields
        st.write("##")
        st.write("##")
        #st.write("write1...")
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg","tif"])
        #st.write(uploaded_file)
        if uploaded_file is not None:
            #st.write(uploaded_file)
            img = Image.open(uploaded_file)
            img.save("test.png")
            img = img.resize((215,90))
            #st.write("write...")
            
            im = tf.keras.utils.load_img('test.png',color_mode='rgb',target_size=(215,90))
            
            input_arr = tf.keras.utils.img_to_array(im)
            input_arr = np.array([input_arr]) 
            #m.predict(input_arr)
            st.image(img)
            #res = np.array(input_arr)
            #res = res.reshape(215,90,3)
            #res = np.expand_dims(res, axis=0)
            #res = np.array(res)
            #st.write(res.shape)
            #res = img.resize((215, 90))
            #res = Image.Resampling.NEAREST(0)
            #st.write("Predicting...")
            result = m.predict(input_arr)
            p = np.argmax(result)
            person = {
                0:'A',
                1: 'B',
                2:'C',
                3:'D',
                4:'E'
            }
            final_result = person[p]
            #st.write(person)

        if st.button("Detect"):
            st.write("Result : " , final_result)
    with right_col:
        st_lottie(animation, height=400, width=350, key='Signature Identification')
