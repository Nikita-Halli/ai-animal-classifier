import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd
import io

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="🐾 AI Animal Breed Classifier",
    page_icon="🐶",
    layout="centered"
)

# ------------------ CSS Styling ------------------
st.markdown("""
<style>
body {
    background-color: #f0f8ff;
}
h1, h2, h3 {
    text-align: center;
    font-family: 'Arial Black', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Title ------------------
st.title("🐾 AI Animal Breed Classifier")
st.write("Upload an image to detect the animal breed using AI.")

# ------------------ Hugging Face Token ------------------
HF_TOKEN = st.secrets["HF_TOKEN"]  # Add your token in Streamlit Secrets

# ------------------ Load Models ------------------
@st.cache_resource
def load_models():
    # Animal classifier
    animal_classifier = pipeline(
        "image-classification",
        model="google/vit-base-patch16-224",
        token=HF_TOKEN
    )
    # Human detection (generic person detection)
    human_detector = pipeline(
        "image-classification",
        model="facebook/deit-base-distilled-patch16-224",
        token=HF_TOKEN
    )
    return animal_classifier, human_detector

animal_classifier, human_detector = load_models()

# ------------------ File Upload ------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=350)

    # ------------------ Human Check ------------------
    with st.spinner("Checking if image contains a human..."):
        human_result = human_detector(image)
        top_label = human_result[0]["label"].lower()
        if "person" in top_label or "human" in top_label:
            st.warning("⚠️ This is a human photo, not an animal!")
        else:
            # ------------------ Animal Prediction ------------------
            with st.spinner("Analyzing animal breed..."):
                results = animal_classifier(image)

            st.success("✅ Prediction Complete!")

            st.subheader("Top 3 Predictions")
            pred_data = {
                "Breed": [r["label"] for r in results[:3]],
                "Confidence (%)": [round(r["score"] * 100, 2) for r in results[:3]]
            }
            df = pd.DataFrame(pred_data)
            st.table(df)

            # ------------------ Download CSV ------------------
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv_buffer.getvalue(),
                file_name="animal_predictions.csv",
                mime="text/csv"
            )
