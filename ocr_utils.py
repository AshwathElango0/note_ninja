import PIL.ImageFile
import fitz
from PIL import Image
import PIL
import numpy as np

def extract_images_from_pdf(pdf_path):
    """Extracts images from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        images = []
        for page_num in range(len(doc)):
            pix = doc[page_num].get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        return images
    except Exception as e:
        raise RuntimeError(f"Error extracting images from PDF: {e}")

def extract_text_with_easyocr(input_data, reader):
    """
    Performs OCR on an image using EasyOCR. 
    Accepts either an image file path or a PIL image object.
    """
    try:
        if isinstance(input_data, Image.Image):  # If input is a PIL Image object
            image = input_data
        elif isinstance(input_data, str):  # If input is a file path
            image = Image.open(input_data)
        else:
            raise ValueError("Input must be a file path (str) or a PIL Image object.")

        # Convert the PIL image to a NumPy array for EasyOCR
        image_np = np.array(image)
        result = reader.readtext(image_np)

        # Extract text and join into a single string
        return " ".join([text[1] for text in result]).strip()
    except Exception as e:
        raise RuntimeError(f"Error during OCR: {e}")

