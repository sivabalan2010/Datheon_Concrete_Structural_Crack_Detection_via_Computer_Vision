from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def generate_report(filename, image_name, edge_percentage):
    pdf = canvas.Canvas(filename, pagesize=A4)

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, 800, "Concrete Inspection Report")

    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, 750, "Project: Concrete Crack Detection")
    pdf.drawString(50, 720, f"Image: {image_name}")
    pdf.drawString(50, 690, f"Detected Edge Pixels: {edge_percentage:.2f}%")

    pdf.drawString(50, 640, "Processing: OpenCV Canny Edge Detection")
    pdf.drawString(50, 610, "Note: This is an educational inspection prototype.")

    pdf.save()
