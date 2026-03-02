import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests

st.title("🌍 Universal Image Classifier")

# Load pretrained model
model = models.mobilenet_v2(pretrained=True)
model.eval()

# ImageNet labels
labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(labels_url).text.split("\n")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top3_prob, top3_catid = torch.topk(probabilities, 3)

    st.subheader("🔎 Predictions:")
    for i in range(3):
        st.write(f"{labels[top3_catid[i]]} ({top3_prob[i].item()*100:.2f}%)")
