import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image
import torchvision.transforms as transforms
import os

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🧠",
    layout="centered"
)

# --------------------------------------------------
# Clean Dark Styling
# --------------------------------------------------
st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    .title {text-align: center; font-size: 40px; font-weight: bold;}
    .subtitle {text-align: center; font-size: 18px; color: #9CA3AF; margin-bottom: 30px;}
    .result-real {color: #22C55E; font-size: 22px; font-weight: bold;}
    .result-fake {color: #EF4444; font-size: 22px; font-weight: bold;}
    .footer {text-align: center; font-size: 13px; color: #6B7280; margin-top: 40px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 Deepfake Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Vision Transformer + Self-Supervised Learning</div>', unsafe_allow_html=True)

device = torch.device("cpu")

# --------------------------------------------------
# Load Model (Fixed Path)
# --------------------------------------------------
@st.cache_resource
def load_model():

    # Get current directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    encoder_path = os.path.join(BASE_DIR, "ssl_encoder.pth")
    classifier_path = os.path.join(BASE_DIR, "classifier.pth")

    encoder = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=0
    )

    encoder.load_state_dict(torch.load(encoder_path, map_location=device))

    class DeepfakeClassifier(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            feature_dim = encoder.num_features
            self.fc = nn.Linear(feature_dim, 1)

        def forward(self, x):
            features = self.encoder(x)
            return self.fc(features)

    model = DeepfakeClassifier(encoder)
    model.load_state_dict(torch.load(classifier_path, map_location=device))
    model.eval()

    return model

model = load_model()

# --------------------------------------------------
# Transform
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# --------------------------------------------------
# Upload Section
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = torch.sigmoid(model(img_tensor))
        confidence = output.item()

    prediction = "Fake" if confidence > 0.5 else "Real"

    st.markdown("---")

    if prediction == "Fake":
        st.markdown(f'<div class="result-fake">Prediction: {prediction}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-real">Prediction: {prediction}</div>', unsafe_allow_html=True)

    st.write(f"Confidence Score: {confidence:.4f}")
    st.progress(float(confidence))

st.markdown('<div class="footer">Built with Vision Transformers & Self-Supervised Learning</div>', unsafe_allow_html=True)