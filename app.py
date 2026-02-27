import streamlit as st
from PIL import Image
from transformers import pipeline
import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
import io

# Page configuration
st.set_page_config(
    page_title="🐶 AI Animal Breed Classifier",
    page_icon="🐾",
    layout="centered"
)

# CSS Styling
st.markdown("""
<style>
body {
    background-color: #f0f8ff;
}
h1, h2, h3 {
    text-align: center;
    font-family: 'Arial Black', sans-serif;
}
.prediction-table {
    margin-left: auto;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🐾 AI Animal Breed Classifier")
st.write("Upload an image to detect the animal breed using AI or detect humans.")

# Load Hugging Face token from secrets
HF_TOKEN = st.secrets["HF_TOKEN"]

# Cache the model to prevent reloading
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224", token=HF_TOKEN)

classifier = load_model()

# Initialize Mediapipe face detector
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Display uploaded image
    st.image(image, caption="Uploaded Image", width=350)
    
    # Convert image to OpenCV format for human detection
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = face_detector.process(image_cv)
    
    if results.detections:
        st.warning("👤 Human detected! Animal breed classification skipped.")
    else:
        # Analyze animal breed
        with st.spinner("Analyzing Image..."):
            predictions = classifier(image)
        
        # Display results
        st.success("✅ Prediction Complete!")
        st.subheader("Top 3 Predictions")
        
        pred_data = {
            "Breed": [r["label"] for r in predictions[:3]],
            "Confidence (%)": [round(r["score"] * 100, 2) for r in predictions[:3]]
        }
        df = pd.DataFrame(pred_data)
        st.table(df)
        
        # Optional: Download results as CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv_buffer.getvalue(),
            file_name="animal_predictions.csv",
            mime="text/csv"
        )
