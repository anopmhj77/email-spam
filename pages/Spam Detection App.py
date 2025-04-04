import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Initialize Porter Stemmer
ps = PorterStemmer()

# Load trained vectorizer and model
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"🚫 Error loading model/vectorizer: {e}")
    st.stop()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters and stopwords
    y = [ps.stem(word) for word in text if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation]
    return " ".join(y)

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("✉️ Enter the message:")

if st.button("🔍 Predict"):
    if input_sms.strip():
        # Preprocess input
        transformed_sms = transform_text(input_sms)

        # Vectorize
        try:
            vector_input = tfidf.transform([transformed_sms])

            # Predict
            result = model.predict(vector_input)[0]

            # Output
            if result == 1:
                st.error("🚨 This is a **Spam** message!")
            else:
                st.success("✅ This is **Not Spam** (Safe Message).")
        except Exception as e:
            st.error(f"🚫 Error in vector transformation: {e}")
    else:
        st.warning("⚠️ Please enter a valid message!")
