import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd

# ------------------- Page Config -------------------
st.set_page_config(page_title="🐶 AI Animal Breed Classifier", 
                   page_icon="🐾", layout="centered")

# ------------------- Custom CSS -------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #fceabb, #f8b500);
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        color: #333;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
    }
    .card {
        background-color: #fff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        margin-top: 20px;
    }
    .prediction:hover {
        background-color: #ffe680;
        border-radius: 8px;
        transition: 0.3s;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- Title -------------------
st.markdown('<div class="title">🐾 Cloud-Based Animal Breed Classifier 🐾</div>', unsafe_allow_html=True)
st.write("Upload an image to detect the animal breed using AI.")

# ------------------- Hugging Face Token -------------------
token = st.text_input("Enter Hugging Face Token", type="password")

# ------------------- Image Upload -------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file and token:
    image = Image.open(uploaded_file)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing Image... 🐶"):
        classifier = pipeline("image-classification", model="google/vit-base-patch16-224", token=token)
        result = classifier(image)

    st.success("✅ Prediction Complete!")

    st.subheader("Top Predictions:")
    predictions_df = pd.DataFrame(result)
    
    for i in result[:3]:
        label = i["label"]
        score = round(i["score"]*100, 2)
        # Add emojis for some common animals
        emoji = "🐶" if "dog" in label.lower() else "🐱" if "cat" in label.lower() else "🐾"
        st.markdown(f'<div class="prediction"> {emoji} **{label}** - {score}% confidence </div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Optional: Download predictions
    predictions_df.to_csv("predictions.csv", index=False)
    st.download_button("Download Predictions CSV", data="predictions.csv", file_name="predictions.csv")
