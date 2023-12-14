import streamlit as st
import os
import numpy as np
from PIL import Image  
from skimage.transform import resize  
import tensorflow as tf
from tensorflow import keras


st.title("Image Classifier using Machine Learning")
st.text("Upload the Image")

# Specify the correct path to your Keras model file (HDF5 format)
model_file_path = '/content/cnnmodel.p'

uploaded_file = st.file_uploader("Choose an image...", type='jpg')

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    # Resize the image
    img_resized = resize(np.array(img), (256, 256))

    st.image(img_resized, caption='Resized Image')

    flat_data = []
    if st.button('PREDICT'):
        flat_data.append(img_resized.flatten())
        flat_data = np.array(flat_data)
        
        # Assuming your model predicts a class index
        class_index = np.argmax(model.predict(flat_data), axis=1)
        print(f'PREDICTED CLASS INDEX: {class_index}')

        # Assuming you have a list of class labels (CATEGORIES)
        # Make sure CATEGORIES is defined appropriately in your code
        # This is just an example; adjust it according to your specific use case
        CATEGORIES = ['class1', 'class2',]
        predicted_class = CATEGORIES[class_index[0]]
        print(f'PREDICTED CLASS: {predicted_class}')
