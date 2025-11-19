import streamlit as st
import numpy as np
import pickle
import fitz
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit.components.v1 as components
from unsupervised import create_topic_model, perform_topic_modeling, kmeans_clustering, plot_clusters
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
import os
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables
load_dotenv()

lemmatizer=WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
st.set_page_config(
    page_title="Document Analysis Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
<style>
    [data-testid="stSidebar"] {display: none;}
    .main-title {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        text-align: center;
        color: #1f77b4;
        margin: 1rem 0 0.5rem 0;
        letter-spacing: 4px;
        line-height: 1.1;
        text-transform: uppercase;
    }
    .sub-title {font-size: 1.2rem; text-align: center; color: #7f8c8d; margin-bottom: 2rem;}
    .control-section {background-color: #1a1a1a; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem;}
    .metric-card {background-color: #1a1a1a; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #333333; box-shadow: 0 2px 4px rgba(0,0,0,0.3); color: #ffffff;}
    .info-box {background-color: #1e3a52; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #3498db; margin: 1rem 0; color: #e8f4f8;}
    .success-box {background-color: #1a3a2a; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #28a745; margin: 1rem 0; color: #d4edda;}
    .section-header {font-size: 3.5rem; font-weight: 600; color: #ffffff; margin: 1.5rem 0 1rem 0; border-bottom: 2px solid #1f77b4; padding-bottom: 0.5rem;}
    .rag-container {background-color: #1a1a1a; padding: 1.5rem; border-radius: 0.5rem; height: 100%;}
</style>
""", unsafe_allow_html=True)

# Initialize ChromaDB client
@st.cache_resource
def init_chromadb():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client

# Initialize Gemini API
@st.cache_resource
def init_gemini():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')
    return model

# Load embedding model for RAG
@st.cache_resource
def load_rag_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# BERT Model Class
class TextClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

def load_resources(model_choice):
    if model_choice == "LSTM":
        model = load_model('lstm_model.keras')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return model, tokenizer, label_encoder
    elif model_choice == "BiLSTM":
        model = load_model('chunk_model.keras')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return model, tokenizer, label_encoder
    elif model_choice == "BERT":
        try:
            with open('pdf_classifier_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            model_config = model_data['model_config']
            label_encoder = model_data['label_encoder']
            bert_tokenizer = model_data['tokenizer']
            
            # Fix tokenizer pad_token issue
            if not hasattr(bert_tokenizer, 'pad_token') or bert_tokenizer.pad_token is None:
                bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            classifier = TextClassifier(
                input_dim=model_config['input_dim'],
                num_classes=model_config['num_classes']
            )
            classifier.load_state_dict(model_data['classifier'])
            classifier.eval()
            
            bert_model = BertModel.from_pretrained('bert-base-uncased')
            bert_model.eval()
            
            return (classifier, bert_model), bert_tokenizer, label_encoder
        except Exception as e:
            st.error(f"Error loading BERT model: {str(e)}")
            return None, None, None
    else:
        return None, None, None
  
def extract_text_from_pdf(pdf_file):
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

def preprocess_text_bert(text):
    """Preprocessing for BERT model"""
    if text is None:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text



def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def predict_document_lstm(text, model, tokenizer, label_encoder):
    preprocessed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded = pad_sequences(sequence, maxlen=1500)
    
    prediction = model.predict(padded, verbose=0)
    predicted_class = np.argmax(prediction[0])
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    
    return predicted_label, prediction[0], label_encoder.classes_

def predict_document_bilstm(text, model, tokenizer, label_encoder):
    preprocessed_text = preprocess_text(text)
    chunks = chunk_text(preprocessed_text, chunk_size=300)
    
    sequences = tokenizer.texts_to_sequences(chunks)
    padded = pad_sequences(sequences, maxlen=300)
    
    chunk_predictions = model.predict(padded, verbose=0)
    avg_prediction = np.mean(chunk_predictions, axis=0)
    
    predicted_class = np.argmax(avg_prediction)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    
    return predicted_label, avg_prediction, label_encoder.classes_

def predict_document_bert(text, models, tokenizer, label_encoder):
    """Predict using BERT model"""
    classifier, bert_model = models
    
    preprocessed_text = preprocess_text_bert(text)
    if not preprocessed_text:
        return "Error", np.array([0]), ["Error"]
    
    # Tokenize with BERT tokenizer
    inputs = tokenizer(
        preprocessed_text,
        return_tensors='pt',
        max_length=512,
        truncation=True,
        padding='max_length'
    )
    
    # Generate BERT embedding
    with torch.no_grad():
        outputs = bert_model(**inputs)
        text_tensor = outputs.last_hidden_state[:, 0, :]
        
        # Classify
        predictions = classifier(text_tensor)
        _, predicted_index = torch.max(predictions, 1)
    
    # Get probabilities (convert from log probabilities)
    probabilities = torch.exp(predictions[0]).numpy()
    predicted_label = label_encoder.inverse_transform(predicted_index.numpy())[0]
    
    return predicted_label, probabilities, label_encoder.classes_

# RAG Functions
def chunk_text_rag(text: str, chunk_size: int = 300, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks with sentence boundary preservation"""
    # Split by sentences first to preserve context
    sentences = re.split(r'[.!?]+\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        words = sentence.split()
        sentence_length = len(words)
        
        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            # Keep overlap by retaining last portion
            overlap_words = ' '.join(current_chunk).split()[-overlap:]
            current_chunk = overlap_words + words
            current_length = len(current_chunk)
        else:
            current_chunk.extend(words)
            current_length += sentence_length
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return [chunk for chunk in chunks if len(chunk.split()) >= 20]

def preprocess_text_rag(text: str) -> str:
    """Clean and preprocess text for RAG while preserving important punctuation"""
    # Remove extra whitespace but preserve paragraph structure
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # Keep important punctuation for context
    text = re.sub(r'[^\w\s.,!?;:()\[\]"-]', '', text)
    # Remove multiple punctuation
    text = re.sub(r'([.,!?;:])+', r'\1', text)
    return text.strip()

def create_vector_store(chunks: List[str], embeddings: np.ndarray, collection_name: str = "pdf_documents"):
    """Create or update ChromaDB collection"""
    client = init_chromadb()
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(embeddings=embeddings.tolist(), documents=chunks, ids=ids)
    return collection

def retrieve_relevant_chunks(query: str, collection, embedding_model, top_k: int = 5) -> List[Tuple[str, float]]:
    """Retrieve and rerank relevant chunks with expanded query"""
    # Expand query with variations for better matching
    query_variations = [query]
    
    # Add keyword extraction from query
    query_words = query.lower().split()
    if len(query_words) > 3:
        key_terms = ' '.join([w for w in query_words if len(w) > 3])
        if key_terms != query.lower():
            query_variations.append(key_terms)
    
    # Retrieve with expanded top_k for reranking
    all_chunks = set()
    all_results = []
    
    for q_var in query_variations:
        query_embedding = embedding_model.encode([q_var])[0]
        results = collection.query(
            query_embeddings=[query_embedding.tolist()], 
            n_results=min(top_k * 2, 20)
        )
        for doc, dist in zip(results['documents'][0], results['distances'][0]):
            if doc not in all_chunks:
                all_chunks.add(doc)
                similarity = 1 - dist
                all_results.append((doc, similarity))
    
    # Sort by similarity and return top_k
    all_results.sort(key=lambda x: x[1], reverse=True)
    return all_results[:top_k]

def generate_answer_with_gemini(query: str, context_chunks: List[Tuple[str, float]], gemini_model) -> str:
    """Generate answer using Gemini with enhanced prompting"""
    # Filter chunks by relevance threshold
    relevant_chunks = [(chunk, sim) for chunk, sim in context_chunks if sim > 0.3]
    
    if not relevant_chunks:
        return "I couldn't find relevant information in the document to answer this question. Please try rephrasing your question or ask about topics covered in the document."
    
    # Build context with better formatting
    context_parts = []
    for i, (chunk, similarity) in enumerate(relevant_chunks, 1):
        context_parts.append(f"**Context Passage {i}** (Relevance: {similarity:.1%}):\n{chunk}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are an expert AI assistant analyzing a document. Your task is to answer questions accurately based on the provided context.

## Document Context:
{context}

## User Question:
{query}

## Instructions:
1. Read all context passages carefully
2. Identify information directly relevant to the question
3. Synthesize a comprehensive answer using ONLY the information from the context
4. If the context provides partial information, explain what is available
5. If the context doesn't contain the answer, clearly state: "The document doesn't contain information about [topic]"
6. Use clear, professional language
7. If relevant, cite which context passage(s) support your answer

## Answer:"""
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating answer: {str(e)}\n\nHowever, here are the most relevant excerpts from the document:\n\n{context}"

def process_classification(raw_text, preprocessed_text, model_choice, model, tokenizer, label_encoder):
    """Thread function for classification"""
    if model_choice == "LSTM":
        return predict_document_lstm(preprocessed_text, model, tokenizer, label_encoder)
    elif model_choice == "BiLSTM":
        return predict_document_bilstm(preprocessed_text, model, tokenizer, label_encoder)
    elif model_choice == "BERT":
        return predict_document_bert(raw_text, model, tokenizer, label_encoder)
    return None, None, None

def process_rag_indexing(raw_text):
    """Thread function for RAG indexing with improved chunking"""
    cleaned_text = preprocess_text_rag(raw_text)
    chunks = chunk_text_rag(cleaned_text, chunk_size=300, overlap=100)
    embedding_model = load_rag_embedding_model()
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    collection = create_vector_store(chunks, embeddings)
    return collection, chunks, embedding_model



st.markdown('<p class="main-title">INTELLIGENT DOCUMENT CLASSIFICATION AND QUERY PROCESSING SYSTEM</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Created by Students of AIDS-B1 Piyush(06419051922), Nikita Babbar(02319051922)</p>', unsafe_allow_html=True)

if 'rag_collection' not in st.session_state:
    st.session_state.rag_collection = None
if 'rag_chunks' not in st.session_state:
    st.session_state.rag_chunks = None
if 'rag_embedding_model' not in st.session_state:
    st.session_state.rag_embedding_model = None
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = init_gemini()
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'results' not in st.session_state:
    st.session_state.results = None

st.markdown('<div class="control-section">', unsafe_allow_html=True)
col_upload, col_model, col_process, col_clear = st.columns([3, 2, 2, 1])
with col_upload:
    uploaded_file = st.file_uploader("Upload PDF Document", type=['pdf'])
with col_model:
    model_choice = st.selectbox(
        "Select Classification Model",
        options=["LSTM", "BiLSTM", "BERT"],
        index=0
    )
with col_process:
    st.write("")
    st.write("")
    process_button = st.button("Start Processing", type="primary", use_container_width=True, disabled=(uploaded_file is None))
with col_clear:
    st.write("")
    st.write("")
    if st.button("Clear Session", use_container_width=True):
        st.session_state.rag_collection = None
        st.session_state.rag_chunks = None
        st.session_state.rag_embedding_model = None
        st.session_state.processed = False
        st.session_state.results = None
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)


if model_choice == "Sequential Clustering":
    st.warning(f"{model_choice} model is not implemented yet. Please choose LSTM, BiLSTM, or BERT.")
    model, tokenizer, label_encoder = None, None, None
else:
    model, tokenizer, label_encoder = load_resources(model_choice)
    if process_button and uploaded_file is not None:
        with st.spinner("Processing document with parallel execution..."):
            raw_text = extract_text_from_pdf(uploaded_file)
            text = raw_text
            preprocessed_text = preprocess_text(raw_text)
            
            if raw_text:
                # Parallel processing using ThreadPoolExecutor
                classification_result = None
                rag_result = None
                
                with ThreadPoolExecutor(max_workers=2) as executor:
                    # Submit both tasks in parallel
                    classification_future = executor.submit(
                        process_classification, raw_text, preprocessed_text, 
                        model_choice, model, tokenizer, label_encoder
                    )
                    rag_future = executor.submit(process_rag_indexing, raw_text)
                    
                    # Wait for both to complete
                    classification_result = classification_future.result()
                    rag_result = rag_future.result()
                
                # Store RAG results in session state
                st.session_state.rag_collection = rag_result[0]
                st.session_state.rag_chunks = rag_result[1]
                st.session_state.rag_embedding_model = rag_result[2]
                
                predicted_label, probabilities, classes = classification_result
                st.session_state.results = {
                    'text': text,
                    'raw_text': raw_text,
                    'predicted_label': predicted_label,
                    'probabilities': probabilities,
                    'classes': classes
                }
                st.session_state.processed = True
                st.markdown('<div class="success-box">Document processed successfully using parallel execution</div>', unsafe_allow_html=True)
    
    if st.session_state.processed and st.session_state.results:
        results = st.session_state.results
        predicted_label = results['predicted_label']
        probabilities = results['probabilities']
        classes = results['classes']
        text = results['text']
        raw_text = results['raw_text']
        
        left_col, right_col = st.columns([1, 1])
        
        with left_col:
                    st.markdown('<p class="section-header">Classification Results</p>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card"><h2 style="margin:0; color:#1f77b4;">{predicted_label.upper()}</h2><p style="margin:0.5rem 0 0 0; color:#7f8c8d;">Predicted Category</p></div>', unsafe_allow_html=True)
                    st.write("")
                    st.markdown("**Confidence Distribution**")
                    for class_name, prob in zip(classes, probabilities):
                        st.progress(float(prob), text=f"{class_name}: {prob*100:.2f}%")
                    st.write("")
                    st.markdown('<p class="section-header">Topic Modeling Results</p>', unsafe_allow_html=True)
                    topic_model_obj, topics, documents, error = perform_topic_modeling(raw_text)
                    if error:
                        st.error(f"Topic modeling error: {error}")
                    else:
                        st.markdown("**Discovered Topics**")
                        topic_info = topic_model_obj.get_topic_info()
                        st.dataframe(topic_info, use_container_width=True)
                        st.markdown("**Top Keywords by Topic**")
                        unique_topics = sorted(set(topics))
                        if -1 in unique_topics:
                            unique_topics.remove(-1)
                        for topic_id in unique_topics[:5]:
                            topic_words = topic_model_obj.get_topic(topic_id)
                            if topic_words:
                                words_str = ", ".join([word for word, _ in topic_words[:10]])
                                st.write(f"**Topic {topic_id}:** {words_str}")
                        st.markdown("**Visualizations**")
                        try:
                            fig1 = topic_model_obj.visualize_topics()
                            st.plotly_chart(fig1, use_container_width=True)
                        except:
                            st.warning("Topic distance map unavailable")
                        try:
                            fig2 = topic_model_obj.visualize_barchart(top_n_topics=min(5, len(unique_topics)))
                            st.plotly_chart(fig2, use_container_width=True)
                        except:
                            st.warning("Bar chart unavailable")
        
        with right_col:
            st.markdown('<div class="rag-container">', unsafe_allow_html=True)
            st.markdown('<p class="section-header">RAG based Question Answering System</p>', unsafe_allow_html=True)
            if st.session_state.rag_collection:
                st.markdown(f'<div class="info-box">Document indexed with {len(st.session_state.rag_chunks)} text chunks</div>', unsafe_allow_html=True)
                query = st.text_input(
                    "Ask a question about the document",
                    placeholder="What is this document about?"
                )
                col_slider, col_btn = st.columns([1, 2])
                with col_slider:
                    top_k = st.slider("Results", 1, 10, 5)
                with col_btn:
                    generate_btn = st.button("Generate Answer", type="primary", use_container_width=True)
                if generate_btn and query:
                    with st.spinner("Processing query..."):
                        relevant_chunks = retrieve_relevant_chunks(
                            query, st.session_state.rag_collection,
                            st.session_state.rag_embedding_model, top_k
                        )
                        st.markdown("**AI Response**")
                        if st.session_state.gemini_model:
                            answer = generate_answer_with_gemini(
                                query, relevant_chunks, st.session_state.gemini_model
                            )
                            st.markdown(answer)
                        else:
                            st.warning("Gemini API not configured")
                            context = "\n\n".join([chunk for chunk, _ in relevant_chunks])
                            st.info(f"Relevant Context:\n\n{context}")
                        st.markdown("**Source References**")
                        for i, (chunk, similarity) in enumerate(relevant_chunks, 1):
                            with st.expander(f"Reference {i} | Similarity: {similarity:.1%}"):
                                st.write(chunk)
                elif generate_btn:
                    st.warning("Please enter a question")
            else:
                st.error("Document indexing failed")
            st.markdown('</div>', unsafe_allow_html=True)
    elif uploaded_file is None:
        st.markdown('<div class="info-box">Please upload a PDF document to begin intelligent analysis</div>', unsafe_allow_html=True)
    elif not st.session_state.processed:
        st.markdown('<div class="info-box">Click "Start Processing" to analyze the document</div>', unsafe_allow_html=True)
        
