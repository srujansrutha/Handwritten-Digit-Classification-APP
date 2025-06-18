import streamlit as st
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np

# Define the model class again
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)

# Load the trained model
model = DigitClassifier()
model.load_state_dict(torch.load("digit_classifier.pth", map_location=torch.device('cpu')))
model.eval()

# Define transformation for user-uploaded image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

st.title("🖊️ Digit Classifier App")
uploaded_file = st.file_uploader("Upload an image of a digit", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = ImageOps.invert(image.convert('L'))  # Convert to grayscale + invert if white on black
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output, 1).item()

    st.subheader(f"🧠 Predicted Digit: **{predicted}**")