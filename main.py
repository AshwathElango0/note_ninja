from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

def extract_text_from_image(image_path):
    """
    Extracts text from an image using the Donut model.

    Args:
        image_path (str): Path to the input image.

    Returns:
        str: Extracted text from the image.
    """
    # Load the Donut processor and model
    processor = DonutProcessor.from_pretrained("nielsr/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("nielsr/donut-base")

    # Open the image
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # Generate text using the model
    outputs = model.generate(pixel_values, max_length=512)

    # Decode the output to text
    extracted_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    return extracted_text

# Example usage
image_path = r"C:\Users\achus\Desktop\epoch_projects\invoice_ocr\data\wordpress-pdf-invoice-plugin-sample_page-0001.jpg"
result = extract_text_from_image(image_path)
print(result)
