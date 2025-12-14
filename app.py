import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --------------------
# Setup
# --------------------
ps = PorterStemmer()

@st.cache_resource
def load_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')

load_nltk_data()

STOP_WORDS = set(stopwords.words('english'))

# --------------------
# Text Preprocessing
# --------------------
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    # Keep alphanumeric tokens
    tokens = [word for word in tokens if word.isalnum()]

    # Remove stopwords and punctuation
    tokens = [
        word for word in tokens
        if word not in STOP_WORDS and word not in string.punctuation
    ]

    # Stemming
    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)

# --------------------
# Load Model & Vectorizer
# --------------------
@st.cache_resource
def load_model():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model

tfidf, model = load_model()

# --------------------
# Streamlit UI
# --------------------
st.title("📩 Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        with st.spinner("Analyzing message..."):
            # Preprocess
            transformed_sms = transform_text(input_sms)

            # Vectorize
            vector_input = tfidf.transform([transformed_sms])

            # Predict
            result = model.predict(vector_input)[0]

        # Display result
        if result == 1:
            st.error("🚨 IT'S A SPAM MESSAGE ")
        else:
            st.success("✅ IT'S A HAM MESSAGE")
