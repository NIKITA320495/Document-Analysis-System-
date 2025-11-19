import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import re
import string
import os
from PyPDF2 import PdfReader
# We no longer need TfidfVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.preprocessing import LabelEncoder
import numpy as np
# NEW: Add imports for transformers
from transformers import BertTokenizer, BertModel

# --- Re-create Model and Helper Functions ---
# IMPORTANT: This class structure MUST match the one you used for training.
# I'm creating a standard simple classifier based on your inputs.
# If your model was different, you must replace this class definition.
class TextClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TextClassifier, self).__init__()
        # UPDATED: Dimensions are now based on the new "size mismatch" error.
        # This should match your trained model exactly.
        self.fc1 = nn.Linear(input_dim, 256) # Was 128
        self.fc2 = nn.Linear(256, 128)        # Was (128, 64)
        self.fc3 = nn.Linear(128, num_classes) # Was (64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # UPDATED: Pass through the three layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # Use LogSoftmax for NLLLoss (which was likely used during training)
        # If you used CrossEntropyLoss, this is still fine for prediction.
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a given PDF file.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Error: PDF file not found at {pdf_path}")
        
    print(f"Extracting text from {pdf_path}...")
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        print("Text extraction successful.")
        return full_text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return None

def preprocess_text(text):
    """
    Cleans and preprocesses text data.
    - Converts to lowercase
    - Removes punctuation
    - Removes numbers
    - Strips extra whitespace
    """
    if text is None:
        return ""
        
    print("Preprocessing text...")
    text = text.lower()  # Lowercase
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove whitespace
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    print("Preprocessing complete.")
    return text

# --- Main Prediction Function ---

def classify_pdf(pdf_path, model_file='pdf_classifier_model.pkl'):
    """
    Loads the saved model and classifies a new PDF file.
    """
    # 1. Check for files
    if not os.path.exists(model_file):
        return f"Error: Model file '{model_file}' not found."
    if not os.path.exists(pdf_path):
        return f"Error: PDF file '{pdf_path}' not found."

    try:
        # 2. Load the saved model data
        print(f"Loading model from {model_file}...")
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)

        # 3. Reconstruct components
        model_config = model_data['model_config']
        label_encoder = model_data['label_encoder']
        
        # This is the BERT tokenizer, not a TF-IDF vectorizer
        bert_tokenizer = model_data['tokenizer'] 
        
        # Initialize the model structure
        classifier = TextClassifier(
            input_dim=model_config['input_dim'],
            num_classes=model_config['num_classes']
        )
        
        # Load the trained weights (state_dict)
        classifier.load_state_dict(model_data['classifier'])
        
        # Set model to evaluation mode (disables dropout, etc.)
        classifier.eval()
        print("Model loaded successfully.")

        # NEW: Load the base BERT model to create embeddings
        # We assume 'bert-base-uncased' because your input_dim is 768.
        # If you used a different model, you must change this string.
        try:
            print("Loading base BERT model (bert-base-uncased)...")
            bert_model = BertModel.from_pretrained('bert-base-uncased')
            bert_model.eval()
        except Exception as e:
            return f"Error loading 'bert-base-uncased'. Do you have internet? Error: {e}"

        # 4. Process the new PDF
        raw_text = extract_text_from_pdf(pdf_path)
        if raw_text is None:
            return "Error: Could not extract text from PDF."
            
        processed_text = preprocess_text(raw_text)
        if not processed_text:
            return "Error: PDF contains no usable text after processing."

        # 5. Vectorize the text -- THIS IS THE NEW WORKFLOW
        print("Tokenizing text with BERT tokenizer...")
        
        # Use the loaded BertTokenizer
        inputs = bert_tokenizer(
            processed_text,
            return_tensors='pt', # Return PyTorch tensors
            max_length=512,      # Standard max length
            truncation=True,
            padding='max_length' # Pad to max length
        )
        
        print("Generating BERT embedding...")
        # Get the BERT model's output
        with torch.no_grad():
            outputs = bert_model(**inputs)
        
        # The embedding for classification is the [CLS] token's output
        # which is the first token in the 'last_hidden_state'
        # This tensor will have the shape [1, 768]
        text_tensor = outputs.last_hidden_state[:, 0, :]

        # 6. Make prediction
        print("Classifying...")
        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = classifier(text_tensor)
            # Get the index of the max log-probability
            _, predicted_index = torch.max(outputs, 1)
            
        # 7. Decode the prediction
        predicted_label_name = label_encoder.inverse_transform(predicted_index.numpy())
        
        return f"The predicted class for '{pdf_path}' is: {predicted_label_name[0]}"

    except Exception as e:
        return f"An error occurred during classification: {e}"

# --- Example Usage ---
if __name__ == "__main__":
    # --- IMPORTANT ---
    # To run this, you must:
    # 1. Have your 'pdf_classifier_model.pkl' file in the same directory.
    # 2. Create a PDF file named 'my_test_pdf.pdf' (or change the path below).
    
    pdf_to_test = r"C:\Users\hp\Desktop\coding\SUPERVISED\testing\try.pdf" 
    
    # Check if a dummy PDF exists, if not, create one for testing
    if not os.path.exists(pdf_to_test):
        print(f"Warning: '{pdf_to_test}' not found.")
        print("Please create this file and add some text to it to test the classifier.")
        # As a placeholder, we'll just show a message.
        # In a real scenario, you would just call classify_pdf.
        print("\n---")
        print("To test, create a file named 'my_test_pdf.pdf' and run: ")
        print(f"python {__file__} {pdf_to_test}")
        print("---")
    else:
        # If the file *does* exist, run the classification
        prediction = classify_pdf(pdf_to_test)
        print("\n--- CLASSIFICATION RESULT ---")
        print(prediction)
        print("-----------------------------")
        
    # Example of how to run it if you pass a file path as an argument
    import sys
    if len(sys.argv) > 1:
        file_from_arg = sys.argv[1]
        print(f"\n--- Classifying file from argument: {file_from_arg} ---")
        prediction = classify_pdf(file_from_arg)
        print("\n--- CLASSIFICATION RESULT ---")
        print(prediction)
        print("-----------------------------")