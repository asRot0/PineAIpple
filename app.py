import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = load_model('models/fruit_cnn_model.h5')

st.title("Fruit Classifier")

uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "png"])
if uploaded_file:
    img = load_img(uploaded_file, target_size=(128, 128))
    st.image(img, caption='Uploaded Image', use_column_width=True)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    st.write(f"Predicted class: {class_idx}")
