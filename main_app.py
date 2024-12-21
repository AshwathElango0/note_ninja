import os
import easyocr
import streamlit as st
from PIL import Image
import pymupdf
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.google import GeminiEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
import numpy as np
import io

# Set up API key for Gemini embedding
google_api_key = "AIzaSyAw786vp_FhAWxi9vce2IoHon53sGxeCdk"
if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = google_api_key

# Initialize LLM models
gemini_model = Gemini(model="models/gemini-1.5-flash", api_key=google_api_key)
embedder = GeminiEmbedding(model_name="models/embedding-001")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # You can add other languages if needed (e.g., ['en', 'de'])

def extract_text_with_easyocr(image):
    """
    Extract text from an image using EasyOCR.
    Converts the image into a format that EasyOCR supports (NumPy array).
    """
    try:
        # Convert PIL Image to NumPy array (which EasyOCR supports)
        image_np = np.array(image)

        # Run EasyOCR text detection
        result = reader.readtext(image_np)

        # Extract the text parts from the result
        extracted_text = " ".join([text[1] for text in result])
        return extracted_text.strip()
    except Exception as e:
        st.error(f"Error during EasyOCR text extraction: {e}")
        return ""

# Function to extract images from PDF pages using PyMuPDF
def extract_images_from_pdf(pdf_path):
    try:
        doc = pymupdf.open(pdf_path)
        images = []
        for page_num in range(len(doc)):
            pix = doc[page_num].get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        return images
    except Exception as e:
        st.error(f"Error extracting images from PDF: {e}")
        return []

# Function to process and index data for RAG
def process_and_index_data(directory, embedder, gemini_model):
    try:
        reader = SimpleDirectoryReader(directory)
        documents = reader.load_data()
        vector_store = VectorStoreIndex.from_documents(documents=documents, **{'embed_model': embedder})
        query_engine = vector_store.as_query_engine(llm=gemini_model)
        return query_engine
    except Exception as e:
        st.error(f"Error building knowledge base: {e}")
        return None

# Streamlit app
st.title("RAG System with Handwritten Notes Support")

# Sidebar file uploader
with st.sidebar:
    st.header("File Uploads")
    uploaded_note_file = st.file_uploader("Upload PDFs or images of handwritten notes", type=['pdf', 'png', 'jpg', 'jpeg'])

# Process uploaded handwritten notes
if uploaded_note_file:
    temp_dir = "./temp/"
    os.makedirs(temp_dir, exist_ok=True)
    temp_note_file_path = os.path.join(temp_dir, f"handwritten_{uploaded_note_file.name}")
    
    # Save the uploaded file locally
    with open(temp_note_file_path, "wb") as f:
        f.write(uploaded_note_file.getbuffer())

    # Select processing method
    st.info("Processing handwritten notes...")
    method = st.radio("Select Text Extraction Method:", ["EasyOCR"], index=0)
    
    # Extract text based on file type
    extracted_text = ""
    if uploaded_note_file.name.lower().endswith(".pdf"):
        images = extract_images_from_pdf(temp_note_file_path)
        for img in images:
            extracted_text += extract_text_with_easyocr(img) + "\n"
    else:
        image = Image.open(temp_note_file_path)
        extracted_text = extract_text_with_easyocr(image)

    # Display extracted text
    st.write("Extracted Text from Handwritten Notes:")
    st.write(extracted_text)

    # Save extracted text for indexing
    data_dir = "./data/"
    os.makedirs(data_dir, exist_ok=True)
    text_file_path = os.path.join(data_dir, f"handwritten_{uploaded_note_file.name}.txt")
    with open(text_file_path, "w") as f:
        f.write(extracted_text)

    # Index and build the knowledge base
    st.info("Adding to knowledge base...")
    query_engine = process_and_index_data(data_dir, embedder, gemini_model)

    if query_engine:
        st.success("Handwritten notes processed and indexed. Ready for querying!")

        # Query interface
        user_query = st.text_input("Ask a question based on handwritten notes:")
        if user_query:
            answer = query_engine.query(user_query)
            st.write("Answer:")
            st.write(answer)
