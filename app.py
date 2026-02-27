import streamlit as st
from PIL import Image
from transformers import pipeline

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(
    page_title="AI Animal Breed Classifier",
    page_icon="🐶",
    layout="centered"
)

st.title("🐶 Cloud-Based Animal Breed Classifier")
st.write("Upload an image to detect the animal breed using AI.")

# ------------------------------
# Load model with caching
# ------------------------------
@st.cache_resource
def load_model(token):
    return pipeline("image-classification", model="google/vit-base-patch16-224", token=token)

# ------------------------------
# Get Hugging Face token
# ------------------------------
# Option 1: Use Secrets (recommended for deployed app)
# Make sure to add HF_TOKEN in Streamlit Secrets
token = st.secrets.get("HF_TOKEN", None)

# Option 2: Use input box (for local testing)
if not token:
    token = st.text_input("Enter Hugging Face Token", type="password")

# ------------------------------
# Upload image
# ------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, width=400, caption="Uploaded Image")

    if token:
        with st.spinner("Analyzing Image..."):
            classifier = load_model(token)
            result = classifier(image)

        st.success("Prediction Complete!")
        st.subheader("Top Predictions:")
        for i in result[:5]:  # top 5 predictions
            st.write(f"🐾 {i['label']} - {round(i['score']*100, 2)}%")
    else:
        st.warning("Please provide your Hugging Face token.")
