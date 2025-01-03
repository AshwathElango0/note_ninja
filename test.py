import os
import easyocr
import shutil
import atexit
import streamlit as st
from PIL import Image
import fitz
import torch
from transformers import pipeline
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.google import GeminiEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.agent import ReActAgent
from llama_index.tools.tavily_research import TavilyToolSpec
from tavily import TavilyClient
from llama_index.core.tools import FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
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

tavily_cli = TavilyClient(api_key=tavily_api_key)

def web_search(query: str) -> str:
    """Function to search the web and obtain information using a search query"""
    results = tavily_cli.search(query=query)
    return results

def mul_integers(a: int, b: int) -> int:
    """Function to multiply 2 integers and return an integer"""
    return a * b

def add_integers(a: int, b: int) -> int:
    """Function to add 2 integers and return an integer"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add_integers)
mul_tool = FunctionTool.from_defaults(fn=mul_integers)
search_tool = FunctionTool.from_defaults(fn=web_search)

buffer = ChatMemoryBuffer(token_limit=10000)
agent = ReActAgent.from_tools(tools=[add_tool, mul_tool, search_tool], llm=gemini_model, memory=buffer)

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
        split_results = executor.map(lambda doc: splitter.split_text(doc.text), documents)
    return [chunk for chunks in split_results for chunk in chunks]

# Build or update vector store
def process_and_index_data(directory, embedder):
    try:
        reader = SimpleDirectoryReader(directory)
        documents = reader.load_data()
        chunks = process_documents_in_parallel(documents, splitter)
        document_chunks = [Document(text=chunk) for chunk in chunks]

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def get_chat_summarizer():
    summarizer = pipeline("summarization", device=device, model="facebook/bart-large-cnn")
    return summarizer

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
        with st.spinner("Processing file..."):
            extracted_text = ""

            if uploaded_note_file.name.lower().endswith(".pdf"):
                with st.spinner("Extracting images from PDF..."):
                    images = extract_images_from_pdf(temp_note_file_path)
                with st.spinner("Performing OCR on extracted images..."):
                    extracted_text = "\n".join([extract_text_with_easyocr(img) for img in images])
            else:
                with st.spinner("Performing OCR on the uploaded image..."):
                    image = Image.open(temp_note_file_path)
                    extracted_text = extract_text_with_easyocr(image)

            with st.spinner("Saving extracted text to file..."):
                text_file_path = os.path.join(DATA_DIR, f"{uploaded_note_file.name}.txt")
                with open(text_file_path, "w") as f:
                    f.write(extracted_text)

            with st.spinner("Indexing text into the knowledge base..."):
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

def recontextualize_query(user_query, conversation_memory):
    # Prepare context from history and retrieved documents
    history_context = "\n".join(
        [f"{role.capitalize()}: {content}" for message in conversation_memory for role, content in message.items()]
    )
    
    # Input prompt for recontextualization
    prompt = (
        f"""The following is the conversation history and relevant information from uploaded files:
        ---
        Conversation History:\n{history_context}
        ---
        User Query: {user_query}
        ---
        Recontextualize the user's query to make it clear, self-contained, and unambiguous."""
    )
    
    # Generate recontextualized query
    response = gemini_model.chat([ChatMessage(content=prompt, role=MessageRole.USER)])
    recontextualized_query = response.message.content
    
    return recontextualized_query

# Chat-based query interface
if st.session_state.retriever:
    user_query = st.text_input("Ask a question based on the uploaded notes:")
    if user_query:
        st.session_state.conversation_memory.append({"user": user_query})

        with st.spinner("Recontextualizing your query..."):
            # Retrieve context
            retrieved_context = st.session_state.retriever.retrieve(user_query)
            
            # Recontextualize user query
            recontextualized_query = recontextualize_query(user_query, st.session_state.conversation_memory)
            st.sidebar.success("Query recontextualized.")

        with st.spinner("Generating response..."):
            # Get response from agent
            answer = agent.chat(message=f"""Context:
                                {' '.join([doc.text for doc in retrieved_context])}
                                Question:
                                {recontextualized_query}""").response
            st.session_state.conversation_memory.append({"assistant": answer})

    # Display chat-like UI
    st.write("Conversation:")
    for message in st.session_state.conversation_memory:
        for role, content in message.items():
            with st.chat_message(role):
                st.write(content)