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
import streamlit.components.v1 as components
from unsupervised import create_topic_model, perform_topic_modeling, kmeans_clustering, plot_clusters
from sentence_transformers import SentenceTransformer

lemmatizer=WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
st.set_page_config(page_title="Document Classification", layout="wide")

def load_resources(model_choice):
    if model_choice == "LSTM":
        model = load_model('lstm_model.keras')
    elif model_choice == "BiLSTM":
        model = load_model('chunk_model.keras')
    else:
        return None, None, None
    
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder
  
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



st.title("Document Analysis")

with st.sidebar:
    st.header("Upload Document")
    
    model_choice = st.selectbox(
        "Choose a model",
        options=["LSTM", "BiLSTM", "BERT"],
        index=0
    )
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    if st.button("Clear"):
        st.rerun()


if model_choice in ["BERT", "Sequential Clustering"]:
    st.warning(f"{model_choice} model is not implemented yet. Please choose LSTM, BiLSTM, or Topic Modeling.")
    model, tokenizer, label_encoder = None, None, None
else:
    model, tokenizer, label_encoder = load_resources(model_choice)
    if uploaded_file is not None:
        with st.spinner("Analyzing document topics..."):
            raw_text = extract_text_from_pdf(uploaded_file)
            text= raw_text
            preprocessed_text = preprocess_text(raw_text)
            
            if raw_text:
                st.success("Document uploaded successfully")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Document Preview")
                    with st.expander("View extracted text"):
                        st.text_area("Text", text[:2000] + "..." if len(text) > 2000 else text, height=300)
                
                with col2:
                    st.subheader("Classification Result")
                    
                    if model_choice == "LSTM":
                        predicted_label, probabilities, classes = predict_document_lstm(
                            preprocessed_text, model, tokenizer, label_encoder
                        )
                    else:
                        predicted_label, probabilities, classes = predict_document_bilstm(
                            preprocessed_text, model, tokenizer, label_encoder
                        )
                    
                    st.metric("Predicted Category", predicted_label.upper())
                    
                    st.subheader("Confidence Scores")
                    for class_name, prob in zip(classes, probabilities):
                        st.progress(float(prob), text=f"{class_name}: {prob*100:.2f}%")
            if raw_text:
                topic_model_obj, topics, documents, error = perform_topic_modeling(raw_text)
                
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.subheader("Topic Modeling Results")
                    
                    topic_info = topic_model_obj.get_topic_info()
                    st.dataframe(topic_info, use_container_width=True)
                    st.subheader("Top Words per Topic")
                    unique_topics = sorted(set(topics))
                    if -1 in unique_topics:
                        unique_topics.remove(-1)
                    
                    for topic_id in unique_topics[:5]:
                        topic_words = topic_model_obj.get_topic(topic_id)
                        if topic_words:
                            words_str = ", ".join([word for word, _ in topic_words[:10]])
                            st.write(f"**Topic {topic_id}:** {words_str}")
                    
                    st.subheader("Visualizations")
                    
                    try:
                        fig1 = topic_model_obj.visualize_topics()
                        
                        st.plotly_chart(fig1, use_container_width=True)
                    except:
                        st.warning("Could not generate topic distance map")
                    
                    try:
                        fig2 = topic_model_obj.visualize_barchart(top_n_topics=min(5, len(unique_topics)))
                        
                        st.plotly_chart(fig2, use_container_width=True)
                    except:
                        st.warning("Could not generate barchart")
                    
                    try:
                        fig3 = topic_model_obj.visualize_hierarchy()
                        
                        st.plotly_chart(fig3, use_container_width=True)
                    except:
                        st.warning("Could not generate hierarchy visualization")

                    st.subheader("K-Means Clustering")
                    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                    embeddings = embedding_model.encode(documents, show_progress_bar=True)
                    num_clusters = min(5, len(set(topics)) - (1 if -1 in topics else 0))
                    labels, centroids = kmeans_clustering(embeddings, num_clusters)
                    fig = plot_clusters(embeddings, labels, centroids, documents)
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Upload a PDF document from the sidebar to perform topic modeling")
        
# elif model is None and model_choice in ["LSTM", "BiLSTM"]:
#     st.warning("Please ensure the following files exist in the directory:")
#     if model_choice == "LSTM":
#         st.code("- lstm_model.keras\n- tokenizer.pkl\n- label_encoder.pkl")
#     else:
#         st.code("- bilstm_model.keras\n- tokenizer.pkl\n- label_encoder.pkl")
# elif model is not None:
#     if uploaded_file is not None:
#         with st.spinner("Processing document..."):
#             text = extract_text_from_pdf(uploaded_file)
#             preprocessed_text = preprocess_text(text)
            
#             if text:
#                 st.success("Document uploaded successfully")
                
#                 col1, col2 = st.columns([2, 1])
                
#                 with col1:
#                     st.subheader("Document Preview")
#                     with st.expander("View extracted text"):
#                         st.text_area("Text", text[:2000] + "..." if len(text) > 2000 else text, height=300)
                
#                 with col2:
#                     st.subheader("Classification Result")
                    
#                     if model_choice == "LSTM":
#                         predicted_label, probabilities, classes = predict_document_lstm(
#                             preprocessed_text, model, tokenizer, label_encoder
#                         )
#                     else:
#                         predicted_label, probabilities, classes = predict_document_bilstm(
#                             preprocessed_text, model, tokenizer, label_encoder
#                         )
                    
#                     st.metric("Predicted Category", predicted_label.upper())
                    
#                     st.subheader("Confidence Scores")
#                     for class_name, prob in zip(classes, probabilities):
#                         st.progress(float(prob), text=f"{class_name}: {prob*100:.2f}%")
#     else:
#         st.info("Upload a PDF document from the sidebar to classify it")
# else:
#     st.info("Select a model and upload a PDF document from the sidebar")