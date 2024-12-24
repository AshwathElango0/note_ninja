import os
import easyocr
import shutil
import atexit
import streamlit as st
from PIL import Image
import pymupdf
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.google import GeminiEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage
import numpy as np
import pickle

# Set up API key for Gemini embedding
google_api_key = "AIzaSyAw786vp_FhAWxi9vce2IoHon53sGxeCdk"
if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = google_api_key

# Initialize LLM models
gemini_model = Gemini(model="models/gemini-1.5-flash", api_key=google_api_key)
embedder = GeminiEmbedding(model_name="models/embedding-001")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # You can add other languages if needed (e.g., ['en', 'de'])

# Function to cache extracted text and vector store
CACHE_DIR = "./cache"  # Global cache directory

def cache_file(file_path):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, os.path.basename(file_path))
    return cache_path


def save_to_cache(obj, cache_path):
    with open(cache_path, 'wb') as cache_file:
        pickle.dump(obj, cache_file)


def load_from_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as cache_file:
            return pickle.load(cache_file)
    return None


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
        st.sidebar.error(f"Error during EasyOCR text extraction: {e}")
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
        st.sidebar.error(f"Error extracting images from PDF: {e}")
        return []


# Function to process and index data for RAG
def process_and_index_data(directory, embedder):
    try:
        reader = SimpleDirectoryReader(directory)
        documents = reader.load_data()
        vector_store = VectorStoreIndex.from_documents(documents=documents, **{'embed_model': embedder})
        return vector_store
    except Exception as e:
        st.sidebar.error(f"Error building knowledge base: {e}")
        return None


# Register cleanup function for cache
def cleanup_cache():
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print(f"Cache directory {CACHE_DIR} cleaned up.")


atexit.register(cleanup_cache)

# Streamlit app
st.title("RAG System with Handwritten Notes Support")

# Sidebar file uploader
with st.sidebar:
    st.header("File Uploads")
    uploaded_note_file = st.file_uploader("Upload PDFs or images of handwritten notes", type=['pdf', 'png', 'jpg', 'jpeg'])

    # Select processing method
    method = st.radio("Select Text Extraction Method:", ["EasyOCR"], index=0)

# Initialize session state for vector store
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if 'retriever' not in st.session_state:
    st.session_state.retriever = None

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = set()

# Process uploaded handwritten notes
if uploaded_note_file:
    temp_dir = "./temp/"
    os.makedirs(temp_dir, exist_ok=True)
    temp_note_file_path = os.path.join(temp_dir, f"handwritten_{uploaded_note_file.name}")

    # Save the uploaded file locally
    with open(temp_note_file_path, "wb") as f:
        f.write(uploaded_note_file.getbuffer())

    # Check if the file has already been processed
    if uploaded_note_file.name not in st.session_state.uploaded_files:
        st.sidebar.info("Processing handwritten notes...")

        # Extract text based on file type
        extracted_text = ""
        if uploaded_note_file.name.lower().endswith(".pdf"):
            images = extract_images_from_pdf(temp_note_file_path)
            for img in images:
                extracted_text += extract_text_with_easyocr(img) + "\n"
        else:
            image = Image.open(temp_note_file_path)
            extracted_text = extract_text_with_easyocr(image)

        # Cache the extracted text
        cache_path = cache_file(temp_note_file_path)
        save_to_cache(extracted_text, cache_path)

        # Save extracted text for indexing
        data_dir = "./data/"
        os.makedirs(data_dir, exist_ok=True)
        text_file_path = os.path.join(data_dir, f"handwritten_{uploaded_note_file.name}.txt")
        with open(text_file_path, "w") as f:
            f.write(extracted_text)

        # Index and build the knowledge base
        st.sidebar.info("Adding to knowledge base...")
        vector_store = process_and_index_data(data_dir, embedder)

        if vector_store:
            st.session_state.vector_store = vector_store
            st.session_state.retriever = vector_store.as_retriever()
            st.sidebar.success("Handwritten notes processed and indexed. Ready for querying!")
            st.session_state.uploaded_files.add(uploaded_note_file.name)
    else:
        st.sidebar.info("File already processed.")

# Query interface
if st.session_state.retriever:
    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = []

    user_query = st.text_input("Ask a question based on handwritten notes:")
    if user_query:
        st.session_state.conversation_memory.append({"user": user_query})

        # Create a prompt to recontextualize the current query
        recontextualization_prompt = [
            ChatMessage(role="system", content="You are an AI assistant helping with handwritten notes. Recontextualize the user's query so it stands independently and removes ambiguity. Include any relevant details from the conversation history. Let it be in the voice of the user, do not make it sound like a summary generated by someone else."),
            ChatMessage(role="user", content=f"Conversation History:\n{st.session_state.conversation_memory}\n\nCurrent Query:\n{user_query}")
        ]
        recontextualized_query = gemini_model.chat(recontextualization_prompt).message.content
        print(recontextualized_query)
        # Retrieve relevant context
        retrieved_context = st.session_state.retriever.retrieve(recontextualized_query)
        context_text = "\n".join([doc.text for doc in retrieved_context])

        # Create input prompt for LLM
        prompt_messages = [
            ChatMessage(role="system", content="You are an AI assistant helping with handwritten notes."),
            ChatMessage(role="user", content=f"Context:\n{context_text}\n\nQuestion:\n{recontextualized_query}")
        ]
        answer = gemini_model.chat(prompt_messages).message.content

        st.session_state.conversation_memory.append({"assistant": answer})

        # Display conversation history
        st.write("Conversation:")
        for message in st.session_state.conversation_memory:
            for role, content in message.items():
                with st.chat_message(role):
                    st.write(content)
