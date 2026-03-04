import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
import matplotlib.pyplot as plt
import json
import datetime

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Animal Breed Classifier", layout="wide")

# -------------------------------
# Simple Login System
# -------------------------------
# You can replace this with a database in production
USER_CREDENTIALS = {
    "admin": "password123",
    "user": "mypassword"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    username = st.session_state.username
    password = st.session_state.password
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        st.session_state.logged_in = True
    else:
        st.error("❌ Invalid username or password")

if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center;'>🔐 Login to Animal Classifier</h1>", unsafe_allow_html=True)
    st.text_input("Username", key="username")
    st.text_input("Password", type="password", key="password")
    st.button("Login", on_click=login)
    st.stop()  # Stop execution until login

# -------------------------------
# Stylish Header
# -------------------------------
st.markdown("""
    <h1 style='text-align: center; color:#4B0082;'>🌍 Animal Breed Image Classifier</h1>
    <p style='text-align: center; font-size:18px; color:gray;'>Upload an image and AI predicts the breed</p>
    <hr style='border:1px solid #eee'>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()
    return model

model = load_model()

# -------------------------------
# Load Labels
# -------------------------------
@st.cache_data
def load_labels():
    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    return requests.get(labels_url).text.split("\n")

labels = load_labels()

# -------------------------------
# Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# Upload Section
# -------------------------------
st.markdown("### 📤 Upload Image")
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], key="uploader")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with st.spinner("🔎 Analyzing Image..."):
        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        top3_prob, top3_catid = torch.topk(probabilities, 3)

    # -------------------------------
    # Display Predictions
    # -------------------------------
    st.markdown("### 🎯 Top Predictions")
    fig, ax = plt.subplots(figsize=(7,4))
    ax.barh(
        [labels[top3_catid[i]] for i in range(3)],
        [top3_prob[i].item() * 100 for i in range(3)],
        color="#4B0082"
    )
    ax.set_xlabel("Confidence (%)")
    ax.invert_yaxis()
    st.pyplot(fig)

    predictions = []
    for i in range(3):
        st.write(f"**{labels[top3_catid[i]]}** — {top3_prob[i].item() * 100:.2f}%")
        predictions.append({
            "label": labels[top3_catid[i]],
            "confidence": round(top3_prob[i].item() * 100, 2)
        })

    # -------------------------------
    # JSON Download
    # -------------------------------
    result = {
        "image_filename": uploaded_file.name,
        "predictions": predictions,
        "timestamp": str(datetime.datetime.now())
    }

    json_data = json.dumps(result, indent=4)
    st.download_button(
        label="📥 Download Predictions as JSON",
        data=json_data,
        file_name="animal_predictions.json",
        mime="application/json"
    )

# -------------------------------
# Model Info
# -------------------------------
st.markdown("---")
st.markdown("### 🧠 Model Information")
st.write("""
- Model: MobileNetV2  
- Framework: PyTorch  
- Dataset: ImageNet  
- Deployment: Streamlit  
- Storage: JSON Download (Local)  
""")
