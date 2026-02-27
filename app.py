import streamlit as st
from PIL import Image
from transformers import pipeline

# Page config (ONLY ONCE)
st.set_page_config(page_title="AI Animal Classifier", page_icon="🐶", layout="centered")

st.markdown("""
    <style>
    .main {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🐶 Cloud-Based Animal Breed Classifier")
st.write("Upload an image to detect the animal breed using AI.")

# Secure token input
token = st.text_input("Enter Hugging Face Token", type="password")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if token:
        classifier = pipeline(
            "image-classification",
            model="google/vit-base-patch16-224",
            token=token
        )

        with st.spinner("Analyzing Image..."):
            result = classifier(image)

        st.success("Prediction Complete!")

        st.subheader("Top 3 Predictions:")

        for i in result[:3]:
            label = i["label"]
            score = round(i["score"] * 100, 2)
            st.write(f"🐾 {label} - {score}% confidence")