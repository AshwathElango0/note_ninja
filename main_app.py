import os
import easyocr
import shutil
import atexit
import streamlit as st
from PIL import Image
import fitz
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.google import GeminiEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.core.llms import ChatMessage
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.agent import ReActAgent
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core.memory import ChatMemoryBuffer
import numpy as np
import pickle
from concurrent.futures import ThreadPoolExecutor

# Constants
CACHE_DIR = "./cache"
TEMP_DIR = "./temp"
DATA_DIR = "./data"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Set up API key for Gemini embedding
google_api_key = "AIzaSyAw786vp_FhAWxi9vce2IoHon53sGxeCdk"
if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = google_api_key

# Initialize models
gemini_model = Gemini(model="models/gemini-1.5-flash", api_key=google_api_key)
embedder = GeminiEmbedding(model_name="models/embedding-001")
splitter = TokenTextSplitter(chunk_size=250, chunk_overlap=50)
reader = easyocr.Reader(['en'])  # EasyOCR Reader

tavily_api_key = "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"
tavily_tool = TavilyToolSpec(api_key=tavily_api_key)
tavily_tool_list = tavily_tool.to_tool_list()

buffer = ChatMemoryBuffer(token_limit=10000)
agent = ReActAgent.from_tools(tools=tavily_tool_list, llm=gemini_model, memory=buffer)  # Initialize ReActAgent

# Caching utilities
def cache_file(file_path):
    return os.path.join(CACHE_DIR, os.path.basename(file_path))

def save_to_cache(obj, cache_path):
    with open(cache_path, 'wb') as cache_file:
        pickle.dump(obj, cache_file)

def load_from_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as cache_file:
            return pickle.load(cache_file)
    return None

# Extract images from PDF pages using PyMuPDF
def extract_images_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        images = []
        for page_num in range(len(doc)):
            pix = doc[page_num].get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        return images
    except Exception as e:
        st.sidebar.error(f"Error extracting images from PDF: {e}")
        return []

# EasyOCR text extraction
def extract_text_with_easyocr(image):
    try:
        image_np = np.array(image)  # Convert PIL image to NumPy array
        result = reader.readtext(image_np)
        return " ".join([text[1] for text in result]).strip()
    except Exception as e:
        st.sidebar.error(f"Error during EasyOCR text extraction: {e}")
        return ""

# Parallel text splitting
def process_documents_in_parallel(documents, splitter):
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Split each document's text into chunks
        split_results = executor.map(lambda doc: splitter.split_text(doc.text), documents)
    # Flatten the results
    return [chunk for chunks in split_results for chunk in chunks]

# Build or update vector store
def process_and_index_data(directory, embedder):
    try:
        reader = SimpleDirectoryReader(directory)
        documents = reader.load_data()
        # Split the documents into chunks
        chunks = process_documents_in_parallel(documents, splitter)
        # Create Document objects from chunks
        document_chunks = [Document(text=chunk) for chunk in chunks]
        # Build the vector store index
        return VectorStoreIndex.from_documents(documents=document_chunks, embed_model=embedder)
    except Exception as e:
        st.sidebar.error(f"Error building knowledge base: {e}")
        return None


def update_vector_store(vector_store, new_documents, embedder):
    try:
        nodes = process_documents_in_parallel(new_documents, splitter)
        new_docs = [Document(**{'text': node.get_content()}) for node in nodes]
        vector_store.add_documents(documents=new_docs, embed_model=embedder)
        return vector_store
    except Exception as e:
        st.sidebar.error(f"Error updating knowledge base: {e}")
        return vector_store

# Cleanup cache on exit
def cleanup_cache():
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)

atexit.register(cleanup_cache)

# Streamlit App
st.title("RAG System with Handwritten Notes Support")

# Sidebar
with st.sidebar:
    st.header("File Uploads")
    uploaded_note_file = st.file_uploader("Upload PDFs or images", type=['pdf', 'png', 'jpg', 'jpeg'])

# Session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = set()
if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = []

# Process uploaded files
if uploaded_note_file:
    temp_note_file_path = os.path.join(TEMP_DIR, uploaded_note_file.name)

    # Save file locally
    with open(temp_note_file_path, "wb") as f:
        f.write(uploaded_note_file.getbuffer())

    if uploaded_note_file.name not in st.session_state.uploaded_files:
        st.sidebar.info("Processing file...")
        extracted_text = ""

        if uploaded_note_file.name.lower().endswith(".pdf"):
            images = extract_images_from_pdf(temp_note_file_path)
            extracted_text = "\n".join([extract_text_with_easyocr(img) for img in images])
        else:
            image = Image.open(temp_note_file_path)
            extracted_text = extract_text_with_easyocr(image)

        text_file_path = os.path.join(DATA_DIR, f"{uploaded_note_file.name}.txt")
        with open(text_file_path, "w") as f:
            f.write(extracted_text)

        if st.session_state.vector_store:
            st.session_state.vector_store = update_vector_store(
                st.session_state.vector_store,
                [Document(**{'text': extracted_text})],
                embedder
            )
        else:
            st.session_state.vector_store = process_and_index_data(DATA_DIR, embedder)

        st.session_state.retriever = st.session_state.vector_store.as_retriever()
        st.session_state.uploaded_files.add(uploaded_note_file.name)
        st.sidebar.success("File processed and indexed.")

# Chat-based query interface
if st.session_state.retriever:
    user_query = st.text_input("Ask a question based on the uploaded notes:")
    if user_query:
        st.session_state.conversation_memory.append({"user": user_query})

        retrieved_context = st.session_state.retriever.retrieve(user_query)
        context_text = "\n".join([doc.text for doc in retrieved_context])
        prompt_messages = [
            ChatMessage(role="system", content="You are an AI assistant helping with handwritten notes."),
            ChatMessage(role="user", content=f"Context:\n{context_text}\n\nQuestion:\n{user_query}"),
        ]
        answer = agent.chat(message=f"Context:\n{context_text}\n\nQuestion:\n{user_query}").response
        st.session_state.conversation_memory.append({"assistant": answer})

    # Display chat-like UI
    st.write("Conversation:")
    for message in st.session_state.conversation_memory:
        for role, content in message.items():
            with st.chat_message(role):
                st.write(content)