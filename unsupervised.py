import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

import re
from bertopic import BERTopic
import nltk
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import plotly.express as px
import plotly.graph_objects as go
stop_words = set(stopwords.words('english'))
lemmatizer=WordNetLemmatizer()

def preprocess_text_for_topic_modeling(text):
    sentences = nltk.sent_tokenize(text)
    cleaned_sentences = []
    for sent in sentences:
        sent = re.sub('[^a-zA-Z]', ' ', sent)
        sent = sent.lower().split()
        sent = [lemmatizer.lemmatize(word) for word in sent if word not in stop_words]
        sent = ' '.join(sent)
        if sent.strip():
            cleaned_sentences.append(sent)
    return cleaned_sentences

def create_topic_model():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 1))
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        verbose=False
    )
    return topic_model

def perform_topic_modeling(text):
    documents = preprocess_text_for_topic_modeling(text)
    
    if len(documents) < 5:
        return None, None, None, "Document too short for topic modeling. Need at least 5 sentences."
    
    topic_model = create_topic_model()
    
    try:
        topics, probs = topic_model.fit_transform(documents)
        return topic_model, topics, documents, None
    except Exception as e:
        return None, None, None, str(e)
    

def kmeans_clustering(X, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return labels, centroids

def plot_clusters(X, labels, centroids, documents):
    """
    Create an interactive Plotly scatter plot for K-Means clustering visualization
    
    Args:
        X: Original embeddings
        labels: Cluster labels for each document
        centroids: Cluster centroids
        documents: List of text documents/sentences
    
    Returns:
        Plotly figure object
    """
    # Reduce dimensionality to 2D using PCA
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    centroids_reduced = pca.transform(centroids)
    
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'PC1': X_reduced[:, 0],
        'PC2': X_reduced[:, 1],
        'Cluster': [f'Cluster {label}' for label in labels],
        'Text': [doc[:100] + '...' if len(doc) > 100 else doc for doc in documents],
        'Full_Text': documents
    })
    
    # Create interactive scatter plot with Plotly
    fig = px.scatter(
        df,
        x='PC1',
        y='PC2',
        color='Cluster',
        hover_data={'PC1': False, 'PC2': False, 'Text': True, 'Full_Text': False, 'Cluster': False},
        title='K-Means Clustering Visualization',
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    # Update hover template for better display
    fig.update_traces(
        marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='white')),
        hovertemplate='<b>Cluster:</b> %{fullData.name}<br>' +
                      '<b>Text:</b> %{customdata[0]}<br>' +
                      '<b>PC1:</b> %{x:.2f}<br>' +
                      '<b>PC2:</b> %{y:.2f}<br>' +
                      '<extra></extra>'
    )
    
    # Add centroids as red X markers
    centroids_df = pd.DataFrame({
        'PC1': centroids_reduced[:, 0],
        'PC2': centroids_reduced[:, 1],
        'Cluster': [f'Centroid {i}' for i in range(len(centroids))]
    })
    
    fig.add_trace(go.Scatter(
        x=centroids_df['PC1'],
        y=centroids_df['PC2'],
        mode='markers',
        marker=dict(
            symbol='x',
            size=15,
            color='red',
            line=dict(width=2, color='darkred')
        ),
        name='Centroids',
        hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>',
        text=centroids_df['Cluster']
    ))
    
    # Update layout for better visualization
    fig.update_layout(
        width=900,
        height=600,
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig

