#!/usr/bin/env python
# coding: utf-8

# In[1]:


from contractions import fix
import re
import streamlit as st
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Load trained model and vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("Tf-idf_Vectorizer.pkl")

# Download required nltk data
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    if not isinstance(text, str) or text is None:
        return ""
    
    # Expand contractions
    text = fix(text)

    # Convert text to lowercase
    text = text.lower()
    
    # Remove mentions, hashtags, and URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtags
    text = re.sub(r'https?://\S+', '', text)  # Remove URLs
    
    # Remove special characters (except apostrophes) and extra spaces
    text = re.sub(r"[^a-zA-Z0-9'\s]", ' ', text)
    text = re.sub(r"\s+", ' ', text).strip()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords and short tokens
    stop_words = set(stopwords.words('english'))
    meaningful_words = [word for word in tokens if len(word) > 1 and word.lower() not in stop_words]
    
    # Return None if there are no meaningful words
    if not meaningful_words:
        return None
    
    return ' '.join(meaningful_words)

# Function to classify email
def classify_email(email_text):
    """Classifies an email as spam or not spam."""
    cleaned_text = clean_text(email_text)
    if not cleaned_text:
        return "⚠️ Not enough content to classify"
    
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vectorized)
    return "🛑 Spam" if prediction == 1 else "✅ Not Spam"

# Streamlit App UI
st.title("📧 Email Spam Classifier")
st.write("Enter an email message below to check if it's spam or not.")

user_email = st.text_area("✉️ Paste your email content here:")

if st.button("Classify Email"):
    if user_email:
        result = classify_email(user_email)
        st.subheader(f"Prediction: {result}")
    else:
        st.warning("Please enter an email message.")





