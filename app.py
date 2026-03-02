import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Pretrained Model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Preprocessing function
def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return np.expand_dims(image, axis=0)

st.title("🌍 Universal Image Classifier")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess(image)
    predictions = model.predict(processed_image)

    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

    st.subheader("🔎 Predictions:")
    for label in decoded:
        st.write(f"{label[1]} ({round(label[2]*100,2)}%)")
