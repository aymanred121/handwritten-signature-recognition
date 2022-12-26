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

st.set_page_config(page_title='Signature Identification', page_icon=':star:')

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.markdown("# Signature Identification :busts_in_silhouette:")



animation = load_lottie('https://assets8.lottiefiles.com/packages/lf20_pzk9h5cf.json')
with st.container():
    left_col, right_col = st.columns(2)
    with left_col:
        ## Input Fields
        st.write("##")
        st.write("##")
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg","tif"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            img = img.resize((1024, 1024))
            st.image(img)

        if st.button("Detect"):
            pass
    with right_col:
        st_lottie(animation, height=400, width=350, key='Signature Identification')
