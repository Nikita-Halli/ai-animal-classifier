import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions

# Load model
model = EfficientNetB0(weights="imagenet")

st.title("🌍 Universal Image Classifier")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    st.write("🔎 Analyzing...")

    predictions = model.predict(img_array)
    decoded = decode_predictions(predictions, top=5)[0]

    st.subheader("📊 Top Predictions:")

    for label in decoded:
        st.write(f"{label[1]} : {round(label[2]*100, 2)}%")
