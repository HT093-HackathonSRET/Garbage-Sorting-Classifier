#from ftplib import parse257
import streamlit as st

from collections import Counter
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model as lm

st.markdown("<h1 style='text-align: center; color: goldrod;'>Garbage Sorting Classifier</h1>",
            unsafe_allow_html=True)

map_dic = {0:'Cardboard', 1: 'Glass', 2: 'Metal', 3: 'Paper', 4: 'Plastic', 5: 'Trash'}
        
def y_pred(F, R, C, a, b): return (R*pow(10, a)) / \
    (1+1j*2*3.14*F*(R*pow(10, a))*(C*pow(10, b)))


def log(x): return np.log(x)/np.log(10)


def save_fig(plt):
    return st.pyplot(plt)


def main_func(img):

    return expensive_computation(img)


def resize_img(img, size):
    img = np.asarray(img.resize((size, size), Image.ANTIALIAS))
    # print(img.shape, type(img))
    return img


@st.cache(suppress_st_warning=True)
def expensive_computation(img):
    st.markdown("<h3 style='text-align: center; color: goldrod;'>MODEL PREDICTION</h3>",
                unsafe_allow_html=True)

   

    m1 = lm("Models//model1.h5")
    p1 = map_dic[m1.predict(np.array([resize_img(img, 224)])).argmax()]
    st.write("CNN Model :", p1)

    #m2 = lm("Models//model2.h5")
    # p2 = map_dic[m2.predict(np.array([resize_img(img, 224)])).argmax()]
    # st.write("CNN 2 (GoogleNet) :", p2) 

    # m3 = lm("Models//model3.h5")
    # p5 = map_dic[m3.predict(np.array([resize_img(img, 224)])).argmax()]
    # st.write("CNN 3:", p5)

   

    # try:
    #     m2 = lm("model1.h5")
    #     p2 = map_dic[m2.predict(np.array([resize_img(img, 124)])).argmax()]
    #     st.write("VGG19 :", p2)
    # except:
    #     pass

    # try:
    #     m6 = lm("efficientnettrans")
    #     im = tf.image.resize(
    #         img, (224, 224), preserve_aspect_ratio=False,
    #         antialias=False, name=None
    #     )
    #     im = m6.predict(np.array([resize_img(im, 224)])).argmax()

    #     p6 = map_dic[im]
    #     st.write("Efficient Net:", p6)
    # except:
    #     pass

    # try:
    #     m7 = lm("Models//sruti//vgg_92_75.h5")
    #     p7 = map_dic[m7.predict(np.array([resize_img(img, 124)])).argmax()]
    #     st.write("VGG16:", p7)
    # except:
    #     pass

    x = Counter([p1])
    x = x.most_common(1)
    # if(len(x)>1):
    #     x = x[0][0]
    # else:
    x = x[0][0]


    st.markdown("<p style='margin-top:1em;font-size : 20px;'>The model has predicted that the image is a <b>"+x+"</b></p>",
                unsafe_allow_html=True)
    # st.write("The model has predicted that the image is a",x)

# Function to Read and Manupilate Images


def load_image(img):
    im = Image.open(img)
    # image = np.array(im)
    return im


# Uploading the File to the Page
uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])

# Checking the Format of the page
if uploadFile is not None:
    # Perform your Manupilations (In my Case applying Filters)
    img = load_image(uploadFile)
    st.image(img)
    st.write("Image Uploaded Successfully")
    main_func(img)
else:
    st.write("Make sure you image is in JPG/PNG Format.")
