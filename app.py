import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Concrete Crack Detection System")

uploaded = st.file_uploader("Upload Concrete Image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    enhanced = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    st.image(img, caption="Original Image")
    st.image(enhanced, caption="Enhanced Image")
    st.image(edges, caption="Detected Edges")

    st.success("Image processing completed!")
