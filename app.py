import streamlit as st
import torch
import matplotlib.pyplot as plt
from model import DigitGeneratorNet  # Import from model.py

# Load model
model = DigitGeneratorNet()
model.load_state_dict(torch.load("digit_generator.pth", map_location="cpu"))
model.eval()

st.title("Handwritten Digit Generator")
digit = st.number_input("Select a digit (0–9):", min_value=0, max_value=9, step=1)

if st.button("Generate Images"):
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        noise = torch.randn(1, 100)
        label = torch.tensor([digit])
        with torch.no_grad():
            img = model(noise, label).squeeze().numpy()
        axs[i].imshow(img, cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)
