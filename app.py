import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd
import io

# -----------------------------
# Page configuration & styling
# -----------------------------
st.set_page_config(
    page_title="🐶 AI Animal Breed Classifier",
    page_icon="🐾",
    layout="centered"
)

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

st.title("🐾 AI Animal Breed Classifier")
st.write("Upload an image to detect the animal breed using AI.")

# -----------------------------
# Load Hugging Face token
# -----------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]  # Add your token in Streamlit Secrets

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224", token=HF_TOKEN)

classifier = load_model()

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display uploaded image
    st.image(image, caption="Uploaded Image", width=350)

    # Analyze the image
    with st.spinner("Analyzing Image..."):
        results = classifier(image)

    # Display results
    st.success("✅ Prediction Complete!")
    st.subheader("Top 3 Predictions")

    # Prepare table
    pred_data = {
        "Breed": [r["label"] for r in results[:3]],
        "Confidence (%)": [round(r["score"] * 100, 2) for r in results[:3]]
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
