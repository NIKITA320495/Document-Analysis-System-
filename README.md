# Intelligent Document Classification and Query Processing System

A sophisticated document analysis platform that combines machine learning classification, topic modeling, and RAG (Retrieval-Augmented Generation) for intelligent PDF document processing and question answering.

**Created by:** Piyush (06419051922) & Nikita Babbar (02319051922) - AIDS-B1

## 🌟 Features

### 📊 Document Classification
- **Multiple Model Support**: Choose between LSTM, BiLSTM, or BERT models for text classification
- **Category Detection**: Automatically classify documents into categories (business, cybersecurity, entertainment, financial, HR, market, politics, scientific publication, sport, tech)
- **Confidence Scores**: View detailed probability distributions across all categories

### 🔍 Topic Modeling
- **Automatic Topic Discovery**: Extract key topics from documents using BERTopic
- **Interactive Visualizations**: View topic distributions with Plotly charts
- **Keyword Extraction**: Identify top keywords for each discovered topic

### 💬 RAG-based Question Answering
- **Intelligent Q&A**: Ask questions about uploaded documents and get AI-generated answers
- **Context-Aware Responses**: Uses Gemini AI for natural language understanding
- **Source References**: View relevant document excerpts with similarity scores
- **ChromaDB Integration**: Efficient vector storage and retrieval

### ⚡ Performance Optimization
- **Parallel Processing**: Simultaneous classification and RAG indexing using ThreadPoolExecutor
- **Smart Chunking**: Overlapping text chunks with sentence boundary preservation
- **Efficient Embeddings**: Cached sentence transformers for fast processing

## 🛠️ Technology Stack

### Machine Learning & AI
- **TensorFlow/Keras**: Deep learning models (LSTM, BiLSTM)
- **PyTorch**: BERT-based classification
- **Transformers**: Hugging Face BERT models
- **BERTopic**: Topic modeling and analysis
- **Sentence Transformers**: Text embeddings
- **Scikit-learn**: Machine learning utilities

### Vector Database & RAG
- **ChromaDB**: Vector database for semantic search
- **Google Gemini AI**: Advanced language model for answer generation

### Web Interface
- **Streamlit**: Interactive web application
- **Plotly**: Interactive data visualizations

### Text Processing
- **NLTK**: Natural language processing
- **PyMuPDF (fitz)**: PDF text extraction
- **Python-dotenv**: Environment variable management

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- Gemini API Key (for RAG functionality)

## 🚀 Installation

### 1. Clone the Repository
```bash
cd c:\Users\Nikita\Desktop\SUPERVISED\SUPERVISED
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

To obtain a Gemini API key:
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Generate and copy your API key

## 📁 Project Structure

```
SUPERVISED/
├── app.py                          # Main Streamlit application
├── ragimplement.py                 # Standalone RAG implementation
├── unsupervised.py                 # Topic modeling and clustering
├── requirements.txt                # Python dependencies
├── lstm_model.keras               # LSTM classification model
├── chunk_model.keras              # BiLSTM classification model
├── lstm_model.ipynb               # LSTM training notebook
├── pdfclassifier.ipynb            # BERT classifier training
├── tokenizer.pkl                  # Text tokenizer
├── label_encoder.pkl              # Label encoder for classes
├── pdf_classifier_model.pkl       # BERT model checkpoint
├── .env                           # Environment variables (create this)
├── bert_doc_classifier_model/     # BERT model files
├── bert_document_classifier/      # Alternative BERT model
├── bert_embeddings/               # Pre-computed embeddings
├── chroma_db/                     # ChromaDB vector storage
└── DATASET/                       # Training dataset
    ├── business/
    ├── cybersecurity/
    ├── entertainment/
    ├── financial/
    ├── HR/
    ├── market/
    ├── politics/
    ├── scientific_publication/
    ├── sport/
    └── tech/
```

## 🎮 Usage

### Running the Main Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Application

1. **Upload Document**: Click "Upload PDF Document" and select your PDF file
2. **Select Model**: Choose between LSTM, BiLSTM, or BERT classification models
3. **Start Processing**: Click "Start Processing" to analyze the document
4. **View Results**: 
   - Classification results with confidence scores
   - Discovered topics with keywords
   - Interactive visualizations
5. **Ask Questions**: Use the RAG Q&A system to query the document
   - Type your question in the input field
   - Adjust the number of results to retrieve (1-10)
   - Click "Generate Answer" to get AI-powered responses

### Standalone RAG Implementation

```bash
streamlit run ragimplement.py
```

A simplified version focused solely on RAG-based question answering.

## 🧠 Models

### LSTM Model
- Sequence-based classification
- Single document processing
- Best for: General document classification

### BiLSTM Model
- Bidirectional LSTM with chunking
- Processes documents in 300-word chunks
- Best for: Long documents with varied content

### BERT Model
- Transformer-based architecture
- Pre-trained on large corpus
- Best for: High-accuracy classification with contextual understanding

## 📊 Supported Document Categories

- **Business**: Corporate documents, reports, memos
- **Cybersecurity**: Security reports, threat analysis
- **Entertainment**: Media, events, reviews
- **Financial**: Financial reports, analysis, market data
- **HR**: Human resources, employment documents
- **Market**: Market research, trends, analysis
- **Politics**: Political news, policy documents
- **Scientific Publication**: Research papers, studies
- **Sport**: Sports news, statistics, reports
- **Tech**: Technology articles, technical documentation

## 🔧 Configuration

### Model Parameters
- **LSTM Max Length**: 1500 tokens
- **BiLSTM Chunk Size**: 300 words
- **BERT Max Length**: 512 tokens
- **RAG Chunk Size**: 300 words
- **RAG Overlap**: 100 words

### RAG Parameters
Adjust in the application:
- **Top K Results**: Number of relevant chunks to retrieve (1-10)
- **Similarity Threshold**: Minimum relevance score (default: 0.3)

## 🎨 Features in Detail

### Parallel Processing
The application uses `ThreadPoolExecutor` to simultaneously:
- Classify documents using selected ML model
- Index document chunks in ChromaDB
- Generate embeddings

This reduces processing time by up to 50%.

### Smart Text Chunking
- Preserves sentence boundaries
- Overlapping chunks for context continuity
- Filters out short, uninformative chunks

### Enhanced Query Processing
- Query expansion for better retrieval
- Relevance scoring and filtering
- Source attribution with similarity scores

## 🐛 Troubleshooting

### Common Issues

**Issue**: `GEMINI_API_KEY not found`
- **Solution**: Ensure `.env` file exists with valid API key

**Issue**: `Model file not found`
- **Solution**: Ensure all `.keras` and `.pkl` files are present

**Issue**: NLTK data missing
- **Solution**: Run NLTK download commands in Installation section

**Issue**: ChromaDB errors
- **Solution**: Delete `chroma_db/` folder and restart application

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size or use CPU-only mode

## 📈 Performance Tips

1. **First Run**: Initial processing may be slower due to model loading
2. **GPU Acceleration**: Install CUDA for faster BERT processing
3. **Memory Management**: Process one document at a time for large files
4. **Cache**: Models and embeddings are cached for subsequent runs

## 🔐 Security Notes

- Store API keys in `.env` file (never commit to version control)
- Add `.env` to `.gitignore`
- Use environment variables for sensitive data

## 📝 Training Your Own Models

### LSTM Model
See `lstm_model.ipynb` for training process

### BERT Model
See `pdfclassifier.ipynb` for training process

Training requires:
- Labeled dataset in `DATASET/` folder
- Sufficient GPU memory (recommended: 8GB+)
- Training time: 1-4 hours depending on dataset size

## 🤝 Contributing

This is an academic project. For questions or suggestions, contact:
- Piyush: 06419051922
- Nikita Babbar: 02319051922

## 📄 License

Academic project for educational purposes.

## 🙏 Acknowledgments

- **BERTopic**: Topic modeling framework
- **Sentence Transformers**: Embedding models
- **ChromaDB**: Vector database
- **Google Gemini**: AI language model
- **Streamlit**: Web application framework

## 📚 References

- [BERTopic Documentation](https://maartengr.github.io/BERTopic/)
- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)
- [Google Gemini AI](https://ai.google.dev/)

## 🔄 Version History

- **v1.0**: Initial release with LSTM, BiLSTM, BERT classification
- **v1.1**: Added RAG-based Q&A system
- **v1.2**: Implemented parallel processing
- **v1.3**: Enhanced UI and visualizations

---

**Made with ❤️ by AIDS-B1 Students**
