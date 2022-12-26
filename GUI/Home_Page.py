import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import requests
import seaborn as sns
from streamlit_lottie import st_lottie


st.set_page_config(page_title='Signature Recognition', page_icon=':star:')

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#st.sidebar.success("Choose one of the following tasks")

st.write("## Handwritten Signature Recognition :writing_hand:")


animation = load_lottie('https://assets4.lottiefiles.com/packages/lf20_pv3vjlpk.json')
with st.container():
    left_col, right_col = st.columns(2)
    with left_col:
       st.markdown(
    """
    Authentication is an important factor to manage security. 
    Signatures are widely used for personal identification and verification.and the first step in the process of signature verification and identification is to detect and extract the signature from the document.
    
    **👈 Select one of Signature Verification, Identification or Detection tasks from the sidebar** to get started.
    
    ### Team Members
    - Ahmed Mohamed Samy 
    - Ayman Hassan
    - Abdelrahman Mohamed
    - Nora Ekramy
    - Nourhan Mahmoud
    ### See More
    - GitHub Repository: [Project](https://github.com/aymanred121/handwritten-signature-recognition)
"""
)
    with right_col:
        st_lottie(animation, height=400, width=350, key='Signature Recognition')
