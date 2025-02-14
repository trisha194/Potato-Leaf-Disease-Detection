import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# Function to download the model if it does not exist
MODEL_URL = "https://drive.google.com/file/d/1pxLcxwRYhyqLVLcG2ShyQ09pgRsUbCTy/view?usp=sharing"
MODEL_PATH = "potato_model.tflite"

if not os.path.exists(MODEL_PATH):
    st.write("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to classify the disease
def classify_disease(image):
    """Classify plant disease using the TFLite model."""
    img = Image.open(image).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data)

# Streamlit App UI
st.set_page_config(page_title="Plant Disease Detection", layout="wide")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose a page", ["Home", "Detect Disease"])

# Display image (Replace 'Diseases.png' with an appropriate path)
st.image("Diseases.png", use_container_width=True)

if page == "Home":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System</h1>", unsafe_allow_html=True)
    st.write("This tool helps in identifying plant diseases using deep learning models.")

elif page == "Detect Disease":
    st.subheader("Upload an image to analyze")
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Predict Disease"):
            st.spinner("Analyzing the image...")
            class_names = ['Potato - Early Blight', 'Potato - Late Blight', 'Potato - Healthy']
            prediction = classify_disease(uploaded_image)
            st.success(f"The model predicts: {class_names[prediction]}")
