import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Universal Image Classifier", layout="centered")

st.markdown("""
    <h1 style='text-align: center;'>🌍 Universal Image Classifier</h1>
    <p style='text-align: center; font-size:18px;'>Upload any image and let AI predict what it sees</p>
""", unsafe_allow_html=True)


# -------------------------------
# Load model (cached for speed)
# -------------------------------
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    return model

model = load_model()


# -------------------------------
# Load ImageNet labels
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


uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with st.spinner("🔎 Analyzing Image..."):
        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        top3_prob, top3_catid = torch.topk(probabilities, 3)

    st.subheader("🎯 Top Predictions")

    # Bar Chart
    fig, ax = plt.subplots()
    ax.barh(
        [labels[top3_catid[i]] for i in range(3)],
        [top3_prob[i].item()*100 for i in range(3)]
    )
    ax.set_xlabel("Confidence (%)")
    ax.invert_yaxis()
    st.pyplot(fig)

    for i in range(3):
        st.write(f"**{labels[top3_catid[i]]}** — {top3_prob[i].item()*100:.2f}%")


# -------------------------------
# Model Info Section
# -------------------------------
st.markdown("---")
st.markdown("### 🧠 Model Information")
st.write("""
- **Model:** MobileNetV2
- **Framework:** PyTorch
- **Dataset:** ImageNet (1000 object categories)
- **Type:** Convolutional Neural Network (CNN)
- **Deployment:** Streamlit Cloud
""")
