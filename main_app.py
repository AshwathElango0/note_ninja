import os
import pytesseract
import streamlit as st
from PIL import Image
import pymupdf  # PyMuPDF
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from transformers import LayoutLMv3Processor, LayoutLMv3Model
import torch
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.google import GeminiEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

# Set up API key for Gemini embedding
google_api_key = "AIzaSyAw786vp_FhAWxi9vce2IoHon53sGxeCdk"
if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = google_api_key

# Initialize LLM models
gemini_model = Gemini(model="models/gemini-1.5-flash", api_key=google_api_key)
embedder = GeminiEmbedding(model_name="models/embedding-001")

# Initialize LayoutLMv3 Processor and Model
layoutlm_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", revision="main")
layoutlm_model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base", revision="main")

# Function to extract text using Tesseract OCR
def extract_text_with_tesseract(image):
    try:
        text = pytesseract.image_to_string(image, lang='eng')
        return text.strip()
    except Exception as e:
        st.error(f"Error during OCR: {e}")
        return ""

def extract_text_with_layoutlm(image):
    """
    Extract text from an image using LayoutLMv3 by leveraging the model's predictions.
    """
    try:
        # Preprocess the input image
        encoding = layoutlm_processor(image, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        input_ids = encoding["input_ids"]
        
        # Forward pass through the model
        outputs = layoutlm_model(**encoding)
        
        # Obtain token predictions (logits)
        logits = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, num_labels)
        
        # Convert logits to predicted IDs
        predicted_ids = torch.argmax(logits, dim=-1)  # Shape: (batch_size, sequence_length)

        # Convert input token IDs to tokens
        tokens = layoutlm_processor.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        # Map predicted IDs to labels (if classification is enabled)
        predicted_labels = [layoutlm_model.config.id2label[pred_id.item()] for pred_id in predicted_ids[0]]

        # Filter out special tokens and corresponding predictions
        filtered_tokens_and_labels = [
            (token, label) 
            for token, label in zip(tokens, predicted_labels)
            if token not in {"[PAD]", "[CLS]", "[SEP]"}
        ]

        # Extract meaningful text based on predictions (Optional: filter by specific labels)
        meaningful_tokens = [token for token, label in filtered_tokens_and_labels if label != "O"]  # 'O' = Outside entity

        # Join meaningful tokens to form extracted text
        extracted_text = " ".join(meaningful_tokens)
        
        return extracted_text.strip()
    except Exception as e:
        st.error(f"Error during LayoutLM text extraction: {e}")
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
    method = st.radio("Select Text Extraction Method:", ["LayoutLM", "Tesseract (OCR)"], index=0)
    
    # Extract text based on file type
    if uploaded_note_file.name.lower().endswith(".pdf"):
        images = extract_images_from_pdf(temp_note_file_path)
        extracted_text = ""
        for img in images:
            if method == "LayoutLM":
                extracted_text += extract_text_with_layoutlm(img) + "\n"
            else:
                extracted_text += extract_text_with_tesseract(img) + "\n"
    else:
        image = Image.open(temp_note_file_path)
        if method == "LayoutLM":
            extracted_text = extract_text_with_layoutlm(image)
        else:
            extracted_text = extract_text_with_tesseract(image)

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
