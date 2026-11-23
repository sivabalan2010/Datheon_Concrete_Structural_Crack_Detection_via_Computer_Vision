
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

st.set_page_config(page_title="Concrete Crack Detection", page_icon="🏗️")

st.title("Concrete Structural Crack Detection")
st.write("Upload a concrete image for image processing and inspection reporting.")

uploaded = st.file_uploader(
    "Upload Concrete Image",
    type=["jpg", "jpeg", "png"]
)

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

    edge_percentage = (np.count_nonzero(edges) / edges.size) * 100

    st.subheader("Image Processing Results")
    st.image(img, caption="Original Image")
    st.image(enhanced, caption="Contrast Enhanced Image")
    st.image(edges, caption="Canny Edge Detection")

    st.metric("Detected Edge Pixels", f"{edge_percentage:.2f}%")

    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=A4)

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, 800, "Concrete Inspection Report")

    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, 765, f"Image: {uploaded.name}")
    pdf.drawString(50, 745, f"Image Size: {image.width} x {image.height}")
    pdf.drawString(50, 725, "Processing: CLAHE and Canny Edge Detection")
    pdf.drawString(50, 705, f"Detected Edge Pixels: {edge_percentage:.2f}%")

    pdf.drawString(50, 670, "Original Concrete Image:")

    original_buffer = BytesIO()
    image.save(original_buffer, format="PNG")
    original_buffer.seek(0)

    pdf.drawImage(
        ImageReader(original_buffer),
        50, 400,
        width=450, height=250,
        preserveAspectRatio=True,
        anchor="c"
    )

    pdf.setFont("Helvetica", 9)
    pdf.drawString(50, 370, "Note: Edge detection identifies image boundaries,")
    pdf.drawString(50, 355, "not confirmed structural cracks.")
    pdf.drawString(50, 340, "This educational report is not a professional")
    pdf.drawString(50, 325, "structural safety assessment.")

    pdf.save()
    pdf_buffer.seek(0)

    st.success("Image processing and PDF report generated!")

    st.download_button(
        label="Download Inspection PDF",
        data=pdf_buffer.getvalue(),
        file_name="Concrete_Inspection_Report.pdf",
        mime="application/pdf"
    )
