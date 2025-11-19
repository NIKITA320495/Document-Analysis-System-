import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
import os
import re
from typing import List, Tuple
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB client - UPDATED
@st.cache_resource
def init_chromadb():
    # Use the new PersistentClient instead of deprecated settings
    client = chromadb.PersistentClient(path="./chroma_db")
    return client

# Initialize Gemini API
@st.cache_resource
def init_gemini():
    """Initialize Gemini API with API key from environment"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
        return None
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')
    return model

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def preprocess_text(text: str) -> str:
    """Clean and preprocess text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def create_vector_store(chunks: List[str], embeddings: np.ndarray, collection_name: str = "pdf_documents"):
    """Create or update ChromaDB collection with document chunks"""
    client = init_chromadb()
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    
    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add documents to collection
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        embeddings=embeddings.tolist(),
        documents=chunks,
        ids=ids
    )
    
    return collection

def retrieve_relevant_chunks(query: str, collection, embedding_model, top_k: int = 5) -> List[Tuple[str, float]]:
    """Retrieve most relevant chunks for a query"""
    # Generate query embedding
    query_embedding = embedding_model.encode([query])[0]
    
    # Search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    # Extract documents and distances
    documents = results['documents'][0]
    distances = results['distances'][0]
    
    # Convert distances to similarity scores (cosine similarity)
    similarities = [1 - dist for dist in distances]
    
    return list(zip(documents, similarities))

def generate_answer_with_gemini(query: str, context_chunks: List[Tuple[str, float]], gemini_model) -> str:
    """Generate answer using Gemini AI based on retrieved context"""
    # Combine relevant chunks
    context = "\n\n".join([f"[Relevance: {similarity:.2%}]\n{chunk}" for chunk, similarity in context_chunks])
    
    # Create prompt for Gemini
    prompt = f"""You are a helpful AI assistant that answers questions based on the provided document context.

Context from the document:
{context}

Question: {query}

Please provide a clear, concise, and accurate answer based ONLY on the information provided in the context above. If the context doesn't contain enough information to answer the question, say so. Do not make up information."""
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating answer: {str(e)}\n\nRelevant context:\n{context}"

def generate_answer_fallback(query: str, context_chunks: List[Tuple[str, float]]) -> str:
    """Fallback answer generation without AI"""
    context = "\n\n".join([chunk for chunk, _ in context_chunks])
    return f"Based on the document, here are the most relevant sections:\n\n{context}"

def main():
    st.set_page_config(page_title="RAG PDF Q&A", layout="wide")
    st.title("📄 RAG-based PDF Question Answering with Gemini AI")
    
    # Initialize session state
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    if 'chunks' not in st.session_state:
        st.session_state.chunks = None
    if 'gemini_model' not in st.session_state:
        st.session_state.gemini_model = init_gemini()
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📤 Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
        
        if uploaded_file is not None:
            if st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    # Extract text
                    text = extract_text_from_pdf(uploaded_file)
                    
                    if text:
                        # Preprocess and chunk
                        cleaned_text = preprocess_text(text)
                        chunks = chunk_text(cleaned_text, chunk_size=500, overlap=50)
                        
                        # Load embedding model
                        embedding_model = load_embedding_model()
                        
                        # Generate embeddings
                        embeddings = embedding_model.encode(chunks, show_progress_bar=True)
                        
                        # Create vector store
                        collection = create_vector_store(chunks, embeddings)
                        
                        # Store in session state
                        st.session_state.collection = collection
                        st.session_state.chunks = chunks
                        st.session_state.embedding_model = embedding_model
                        
                        st.success(f"✅ PDF processed! Created {len(chunks)} chunks.")
        
        if st.button("Clear Database"):
            st.session_state.collection = None
            st.session_state.chunks = None
            st.rerun()
    
    # Main content area
    if st.session_state.collection is None:
        st.info("👈 Please upload a PDF from the sidebar to get started")
    else:
        st.success(f"📚 Document loaded with {len(st.session_state.chunks)} chunks")
        
        # Question input
        st.subheader("❓ Ask a Question")
        query = st.text_input("Enter your question about the document:", 
                             placeholder="e.g., What is the main topic of this document?")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            top_k = st.slider("Results to retrieve:", min_value=1, max_value=10, value=5)
        
        if st.button("Get Answer", type="primary"):
            if query:
                with st.spinner("Searching for answer..."):
                    # Retrieve relevant chunks
                    relevant_chunks = retrieve_relevant_chunks(
                        query, 
                        st.session_state.collection,
                        st.session_state.embedding_model,
                        top_k=top_k
                    )
                    
                    # Generate answer with Gemini
                    st.subheader("🤖 AI-Generated Answer")
                    
                    if st.session_state.gemini_model:
                        with st.spinner("Generating answer with Gemini AI..."):
                            answer = generate_answer_with_gemini(query, relevant_chunks, st.session_state.gemini_model)
                            st.markdown(answer)
                    else:
                        st.warning("⚠️ Gemini API key not provided. Showing basic results.")
                        answer = generate_answer_fallback(query, relevant_chunks)
                        st.info(answer)
                    
                    # Show relevant chunks with similarity scores
                    st.subheader("📝 Source Context")
                    for i, (chunk, similarity) in enumerate(relevant_chunks, 1):
                        with st.expander(f"Source {i} (Similarity: {similarity:.2%})"):
                            st.write(chunk)
            else:
                st.warning("Please enter a question")
        
        # Display document statistics
        with st.expander("📊 Document Statistics"):
            st.metric("Total Chunks", len(st.session_state.chunks))
            st.metric("Average Chunk Length", 
                     f"{sum(len(c.split()) for c in st.session_state.chunks) / len(st.session_state.chunks):.0f} words")

if __name__ == "__main__":
    main()