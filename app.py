import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def classify_disease(image_path):
    model = tf.keras.models.load_model("potato_disease_model.h5")
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    return np.argmax(predictions)

# Streamlit App UI
st.set_page_config(page_title="Plant Disease Detection", layout="wide")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose a page", ["Home", "Detect Disease"])

st.image("Diseases.png", use_column_width=True)

if page == "Home":
    st.markdown("""<h1 style='text-align: center;'>Plant Disease Detection System""", unsafe_allow_html=True)
    st.write("This tool helps in identifying plant diseases using deep learning models.")

elif page == "Detect Disease":
    st.subheader("Upload an image to analyze")
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Predict Disease"):
            st.spinner("Analyzing the image...")
            class_names = ['Potato - Early Blight', 'Potato - Late Blight', 'Potato - Healthy']
            prediction = classify_disease(uploaded_image)
            st.success(f"The model predicts: {class_names[prediction]}")